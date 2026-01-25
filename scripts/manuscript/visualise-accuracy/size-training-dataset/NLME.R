library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(nlme)
library(ggplot2)

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
    log_rel_wis = log(relative_WIS_nodrift),
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

fixed_effects <- list(
  mu    ~ 1 + season + model + ED,  # linear baseline
  Delta ~ 0 + model,         # total training gain
  kappa ~ 0 + model                 # learning speed
)

random_effects <- pdDiag(mu ~ 1)

start_vals <- c(
  rep(-0.3, nlevels(df$season)),
  rep(0.0,  nlevels(df$model) - 1),     
  -0.05,
  rep(0.2,  nlevels(df$model)),
  rep(0.2,  nlevels(df$model))       
)

names(start_vals) <- c(
  paste0("mu.season", levels(df$season)),
  paste0("mu.model",  levels(df$model)[-1]),
  "mu.ED",
  paste0("Delta.", levels(df$model)),
  paste0("kappa.", levels(df$model))
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
    pnlsMaxIter = 500,
    tolerance = 1e-9,
    msVerbose = TRUE
  )
)

summary(m1)

plot(fitted(m1), resid(m1))

beta <- fixef(m1)

#####################
## Goodness-of-fit ##
#####################

# define grid levels
ED_levels    <- c(0, 1)
ED_labels    <- c("No ED visits", "ED visits")
model_levels <- c("SIR-1S", "SIR-2S", "SIR-3S")
season_levels <- levels(factor(df$season))

# create a prediction grid
pred_grid <- expand.grid(
  training_horizon = sort(unique(df$training_horizon)),
  model            = model_levels,
  ED               = ED_levels,
  season           = season_levels,
  KEEP.OUT.ATTRS   = FALSE,
  stringsAsFactors = FALSE
)
pred_grid <- pred_grid %>%
  mutate(
    ED_num = ED,  # numeric 0/1 for linear predictor
    ED     = factor(ED, levels = ED_levels, labels = ED_labels),
    season = factor(season, levels = season_levels),
    model  = factor(model, levels = model_levels)
  )

# reconstruct mean prediction
pred_grid <- pred_grid %>%
  mutate(
    ## Baseline mean (mu)
    mu =
      beta["mu.(Intercept)"] +
      ifelse(season == "2024-2025",
             beta["mu.season2024-2025"], 0) +
      ifelse(model == "SIR-2S",
             beta["mu.modelSIR-2S"], 0) +
      ifelse(model == "SIR-3S",
             beta["mu.modelSIR-3S"], 0) +
      beta["mu.ED"] * ED_num,
    
    ## Model-specific learning amplitude
    Delta = case_when(
      model == "SIR-1S" ~ beta["Delta.modelSIR-1S"],
      model == "SIR-2S" ~ beta["Delta.modelSIR-2S"],
      model == "SIR-3S" ~ beta["Delta.modelSIR-3S"]
    ),
    
    ## Model-specific learning rate
    kappa = case_when(
      model == "SIR-1S" ~ beta["kappa.modelSIR-1S"],
      model == "SIR-2S" ~ beta["kappa.modelSIR-2S"],
      model == "SIR-3S" ~ beta["kappa.modelSIR-3S"]
    ),
    
    ## Predicted outcome
    log_rel_wis_hat = mu + Delta * exp(-kappa * training_horizon),
    rel_wis_hat     = exp(log_rel_wis_hat)
  )


# build dataframe for visualisation
ED_levels <- c(0, 1)
ED_labels <- c("No ED visits", "ED visits")
plot_df <- df %>%
  mutate(
    rel_wis = relative_WIS_nodrift,
    ED = factor(ED, levels = ED_levels, labels = ED_labels),
    season = factor(season)
  )

# compute the geometric mean of the data and model
obs_gm <- plot_df %>%
  group_by(
    training_horizon,
    model,
    season,
    ED
  ) %>%
  summarise(
    gm_rel_wis = exp(mean(log(rel_wis), na.rm = TRUE)),
    .groups = "drop"
  )
pred_gm <- pred_grid %>%
  group_by(
    training_horizon,
    model,
    season,
    ED
  ) %>%
  summarise(
    gm_rel_wis_hat = exp(mean(log(rel_wis_hat))),
    .groups = "drop"
  )

# force levels to be the same
ED_levels    <- levels(obs_gm$ED)
season_levels <- levels(obs_gm$season)
pred_gm <- pred_gm %>%
  mutate(
    ED = factor(ED, levels = ED_levels),
    season = factor(season, levels = season_levels)
  )


# visualise
ggplot() +
  geom_point(
    data = obs_gm,
    aes(
      x = training_horizon,
      y = gm_rel_wis,
      color = model
    ),
    size = 2,
    alpha = 0.8
  ) +
  geom_line(
    data = pred_gm,
    aes(
      x = training_horizon,
      y = gm_rel_wis_hat,
      color = model
    ),
    linewidth = 0.5,
    alpha=1
  ) +
  facet_grid(
    ED ~ season
  ) +
  scale_y_continuous(
    name = "Geometric mean relative WIS",
    limits = c(0, NA)
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
