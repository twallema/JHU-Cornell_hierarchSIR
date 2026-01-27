library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(nlme)
library(ggplot2)
library(performance)

##################################################
## Set working directory and load accuracy data ##
##################################################

# Helper function to find working directory
get_script_dir <- function() {
  # Case 1: RStudio
  if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
    path <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(path)) return(dirname(path))
  }
  # Case 2: Rscript / source()
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  # Fallback
  getwd()
}

script_dir <- get_script_dir()
data <- read_csv(file.path(script_dir, "accuracy.csv"))

######################
## Prepare the data ##
######################

df <- data %>%
  mutate(
    training_horizon = as.integer(str_sub(hyperparameters, -1)),
    log_rel_wis = log( relative_WIS_nodrift),
    ED = as.numeric(ED_visits == TRUE),
    model = factor(model),
    season = factor(season),
    reference_date = factor(reference_date)
  )

######################
## Define the model ##
######################

nlme_formula <- log_rel_wis ~
  mu +
  Delta * exp(-kappa * training_horizon)

# Consider adding ED to Delta too
fixed_effects <- list(
  mu    ~ 1 + model + season + ED, # why not model? --> doing so pushes kappa.SIR1S to zero (quasi-linear learning) with a way too low asymptote --> unrealistic
  Delta ~ 0 + model + season + ED, # season-specific asymptote
  kappa ~ 0 + model + ED           # shared learning rate per model
)

random_effects <- pdDiag(mu ~ 1)

start_vals <- c(
  ## ---- mu ----
  -0.5,                                  # mu.(Intercept)
  rep(0.1, nlevels(df$season) - 1),      # mu.season*
  rep(0.1, nlevels(df$model) - 1),       # mu.model*
  -0.1,                                  # mu.ED
  
  ## ---- Delta ----
  rep(0.2, nlevels(df$model)),           # Delta.model*
  rep(0.2, nlevels(df$season)-1),        # Delta.season*
  -0.1,                                  # Delta.ED
  
  ## ---- kappa ----
  rep(0.2, nlevels(df$model)),           # kappa.model*
  -0.1                                   # kappa.ED
)


names(start_vals) <- c(
  ## ---- mu ----
  "mu.(Intercept)",
  paste0("mu.season", levels(df$season)[-1]),
  paste0("mu.model", levels(df$model)[-1]),
  "mu.ED",
  
  ## ---- Delta ----
  paste0("Delta.model", levels(df$model)),
  paste0("Delta.season", levels(df$season)[-1]),
  "Delta.ED",
  
  ## ---- kappa ----
  paste0("kappa.model", levels(df$model)),
  "kappa.ED"
)


###################
## Fit the model ##
###################

m1 <- nlme(
  model  = nlme_formula,
  data   = df,
  fixed  = fixed_effects,
  random = random_effects,
  groups = ~ reference_date,
  start  = start_vals,
  na.action = na.exclude,
  control = nlmeControl(
    maxIter = 1000,
    pnlsMaxIter = 1000,
    tolerance = 1e-9
  )
)

# parameters
summary(m1)

# plot residual (lacks structure/correlation?)
plot(fitted(m1), resid(m1))

# verify normality of residuals
qqnorm(resid(m1))
qqline(resid(m1))

# does residual depend on training horizon?
boxplot(resid(m1) ~ df$training_horizon)

# do the random effects satisfy the assumption of normality?
re <- ranef(m1)
qqnorm(re[, 1], main = "QQ plot of reference_date random intercepts")
qqline(re[, 1])

# save fixed effects
beta <- fixef(m1)

# compute average 'Delta' over seasons
season_weights <- prop.table(table(df$season))
Delta_bar <- sapply(levels(df$model), function(m) {
  sum(season_weights * c(
    beta[paste0("Delta.model", m, ":season2023-2024")],
    beta[paste0("Delta.model", m, ":season2024-2025")]
  ))
})

#####################
## Goodness-of-fit ##
#####################

# levels
ED_levels     <- c(0, 1)
ED_labels     <- c("No ED visits", "ED visits")
model_levels  <- c("SIR-1S", "SIR-2S", "SIR-3S")
season_levels <- levels(factor(df$season))

# prediction grid
pred_grid <- expand.grid(
  training_horizon = sort(unique(df$training_horizon)),
  model  = model_levels,
  ED     = ED_levels,
  season = season_levels,
  KEEP.OUT.ATTRS = FALSE
) %>%
  mutate(
    ED_num = ED,
    ED     = factor(ED, levels = ED_levels, labels = ED_labels),
    model  = factor(model, levels = model_levels),
    season = factor(season, levels = season_levels)
  )

# reconstruct model prediction
pred_grid <- pred_grid %>%
  mutate(
    ## baseline mean
    mu =
      beta["mu.(Intercept)"] +
      ifelse(season == "2024-2025", beta["mu.season2024-2025"], 0) +
      ifelse(model == "SIR-2S", beta["mu.modelSIR-2S"], 0) +
      ifelse(model == "SIR-3S", beta["mu.modelSIR-3S"], 0) +
      beta["mu.ED"] * ED_num,
    
    ## season × model learning amplitude
    Delta =
      ifelse(season == "2024-2025", beta["Delta.season2024-2025"], 0) +
      ifelse(model == "SIR-2S", beta["Delta.modelSIR-2S"], 0) +
      ifelse(model == "SIR-3S", beta["Delta.modelSIR-3S"], 0) +
      beta["Delta.ED"] * ED_num,
    
    ## model-specific learning rate + ED shift
    kappa =
      case_when(
        model == "SIR-1S" ~ beta["kappa.modelSIR-1S"],
        model == "SIR-2S" ~ beta["kappa.modelSIR-2S"],
        model == "SIR-3S" ~ beta["kappa.modelSIR-3S"]
      ) +
      beta["kappa.ED"] * ED_num,
    
    log_rel_wis_hat = mu + Delta * exp(-kappa * training_horizon),
    rel_wis_hat     = exp(log_rel_wis_hat)
  )

# observed geometric means
obs_gm <- df %>%
  mutate(
    ED = factor(ED, levels = ED_levels, labels = ED_labels)
  ) %>%
  group_by(training_horizon, model, season, ED) %>%
  summarise(
    gm_rel_wis = exp(mean(log( relative_WIS_nodrift), na.rm = TRUE)),
    .groups = "drop"
  )

# predicted geometric means (already deterministic, but kept symmetric)
pred_gm <- pred_grid %>%
  group_by(training_horizon, model, season, ED) %>%
  summarise(
    gm_rel_wis_hat = exp(mean(log(rel_wis_hat))),
    .groups = "drop"
  )

# plot
p <- ggplot() +
  geom_point(
    data = obs_gm,
    aes(x = training_horizon, y = gm_rel_wis, color = model, shape = model),
    size = 2,
    alpha = 1,
  ) +
  geom_line(
    data = pred_gm,
    aes(x = training_horizon, y = gm_rel_wis_hat, color = model),
    linewidth = 0.5,
    alpha = 0.7
  ) +
  facet_grid(ED ~ season) +
  scale_y_continuous(
    name = "Rel. WIS (GRW)",
    limits = c(0.3, 0.7)
  ) +
  scale_x_continuous(
    name = "Number of training seasons"
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey95"),
    panel.grid.minor = element_blank()
  )
ggsave(
  filename = file.path(script_dir, "training_GRW.pdf"),
  plot = p,
  width = 8.3,
  height = 11.7/3,
  units = "in"
)
