library(readr)
library(ggplot2)
library(dplyr)
library(emmeans)
library(lmerTest)
library(purrr)
library(tidyr)
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
df <- read_csv(file.path(script_dir, "accuracy.csv"))

######################################
## Bootstrap geometric mean rel WIS ##
######################################

# for informed == TRUE
df_boot <- df %>%
  mutate(
    informed = factor(informed),
    model = factor(model, levels = c("SIR-1S", "SIR-2S", "SIR-3S")),
    immunity_linking = factor(immunity_linking, levels = c(FALSE, TRUE)),
    ED_visits = factor(ED_visits, levels = c(FALSE, TRUE)),
    log_rel_wis = log(relative_WIS_drift)
  )

boot_geom_mean <- function(x, B = 2000) {
  n <- length(x)
  boot_stats <- replicate(
    B,
    {
      sample_x <- sample(x, size = n, replace = TRUE)
      mean(sample_x, na.rm = TRUE)
    }
  )
  tibble(
    geom_mean_WIS = exp(mean(x, na.rm = TRUE)),
    lower_95  = exp(quantile(boot_stats, 0.025, na.rm = TRUE)),
    upper_95  = exp(quantile(boot_stats, 0.975, na.rm = TRUE))
  )
}

boot_tbl <- df_boot %>%
  group_by(informed, model, immunity_linking, ED_visits) %>%
  summarise(
    stats = list(boot_geom_mean(log_rel_wis)),
    .groups = "drop"
  ) %>%
  unnest(stats)

boot_tbl_out <- boot_tbl %>%
  mutate(
    geom_mean_WIS = round(geom_mean_WIS, 2),
    lower_95  = round(lower_95, 2),
    upper_95  = round(upper_95, 2)
  ) %>%
  arrange(informed, model, immunity_linking, ED_visits)

write.csv(
  boot_tbl_out,
  file = file.path(script_dir, "bootstrapped_relative_WIS_GRWD.csv"),
  row.names = FALSE
)

# basic point + errorbar plot for informed==TRUE
p <- ggplot(boot_tbl %>% filter(informed==TRUE), aes(x = model, y = geom_mean_WIS, color = ED_visits)) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbar(aes(ymin = lower_95, ymax = upper_95), 
                width = 0.2, 
                position = position_dodge(width = 0.5)) +
  facet_wrap(~ immunity_linking, labeller = labeller(immunity_linking = c("FALSE" = "(a) No immunity chaining", "TRUE" = "(b) Immunity chaining"))) +
  theme_minimal(base_size = 14) +
  labs(
    y = "Rel. WIS (GRW)",
    x = "No. Strains Represented",
    color = "ED visits",
  ) +
  scale_x_discrete(
    labels = c(
      "SIR-1S" = "1",
      "SIR-2S" = "2",
      "SIR-3S" = "3"
    )
  ) + 
  scale_color_brewer(palette = "Set1")
ggsave(
  filename = file.path(script_dir, "WIS_comparison_GRW.pdf"),
  plot = p,
  width = 8.3,
  height = 11.7/4,
  units = "in"
)

#############################################
## Prove informed==FALSE << informed==TRUE ##
#############################################

# compute difference in geom_mean between informed FALSE/TRUE per reference date
paired_df <- df %>%
  filter(informed %in% c(TRUE, FALSE)) %>%
  group_by(reference_date, informed) %>%
  summarise(
    log_gm_WIS = mean(log(relative_WIS_nodrift), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  tidyr::pivot_wider(
    names_from = informed,
    values_from = log_gm_WIS,
    names_prefix = "informed_"
  ) %>%
  filter(!is.na(informed_TRUE), !is.na(informed_FALSE)) %>%
  mutate(
    log_diff = informed_TRUE - informed_FALSE
  )

# bootstrap them
B <- 10000
boot_log_diff <- replicate(
  B,
  {
    idx <- sample(seq_len(nrow(paired_df)), replace = TRUE)
    mean(paired_df$log_diff[idx])
  }
)

# report results
ci <- quantile(boot_log_diff, c(0.025, 0.975))
result <- tibble(
  log_ratio = mean(boot_log_diff),
  lower_log = ci[1],
  upper_log = ci[2],
  diff_percent = (exp(mean(boot_log_diff))-1)*100,
  lower = (exp(ci[1])-1)*100,
  upper = (exp(ci[2])-1)*100
)
result

############################################################
## Prove immunity linking doesn't work for informed==TRUE ##
############################################################

# I don't think we have to prove that it doesn't work for the groups ED_visits and model
# Folks will be able to see on the figure immunity_linking never achieves lower WIS
# It only barely achieves similar WIS scores to having no immunity linking

# compute paired difference between groups
paired_immune <- df %>%
  filter(
    informed == TRUE,
    immunity_linking %in% c(TRUE, FALSE)
  ) %>%
  group_by(reference_date, immunity_linking) %>%
  summarise(
    log_gm_WIS = mean(log(relative_WIS_nodrift), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = immunity_linking,
    values_from = log_gm_WIS,
    names_prefix = "immune_"
  ) %>%
  filter(!is.na(immune_TRUE), !is.na(immune_FALSE)) %>%
  mutate(
    log_diff = immune_TRUE - immune_FALSE
  )

# perform paired bootstrap
B <- 10000
boot_log_diff <- replicate(
  B,
  {
    idx <- sample(seq_len(nrow(paired_immune)), replace = TRUE)
    mean(paired_immune$log_diff[idx])
  }
)

# report results
ci <- quantile(boot_log_diff, c(0.025, 0.975))
result <- tibble(
  log_ratio = mean(boot_log_diff),
  lower_log = ci[1],
  upper_log = ci[2],
  ratio = (exp(mean(boot_log_diff))-1)*100,
  lower = (exp(ci[1])-1)*100,
  upper = (exp(ci[2])-1)*100
)
result

# p-value
p_value <- mean(boot_log_diff <= 0)

##########################################################################
## Prove ED_visits works for informed==TRUE and immunity_linking==FALSE ##
##########################################################################

# compute paired difference between groups (add model == 'SIR-1S' to test model-by-model ED_visits works)
paired_ED_visits <- df %>%
  filter(
    informed == TRUE,
    immunity_linking == FALSE,
  ) %>%
  group_by(reference_date, ED_visits) %>%
  summarise(
    log_gm_WIS = mean(log(relative_WIS_drift), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = ED_visits,
    values_from = log_gm_WIS,
    names_prefix = "ED_visits_"
  ) %>%
  filter(!is.na(ED_visits_TRUE), !is.na(ED_visits_FALSE)) %>%
  mutate(
    log_diff = ED_visits_TRUE - ED_visits_FALSE
  )

# perform paired bootstrap
B <- 10000
boot_log_diff <- replicate(
  B,
  {
    idx <- sample(seq_len(nrow(paired_immune)), replace = TRUE)
    mean(paired_ED_visits$log_diff[idx])
  }
)

# report results
ci <- quantile(boot_log_diff, c(0.025, 0.975))
result <- tibble(
  log_ratio = mean(boot_log_diff),
  lower_log = ci[1],
  upper_log = ci[2],
  ratio = (exp(mean(boot_log_diff))-1)*100,
  lower = (exp(ci[1])-1)*100,
  upper = (exp(ci[2])-1)*100
)
result

# p-value
p_value <- mean(boot_log_diff >= 0)

##################################################################################################
## Prove SIR-1S < SIR-2S < SIR-3S for informed==TRUE, immunity_linking==FALSE, ED_visits==FALSE ##
## Prove SIR-1S < SIR-2S = SIR-3S for informed==TRUE, immunity_linking==FALSE, ED_visits==TRUE  ##
##################################################################################################

# compute paired difference between models (1 on 2, 1 on 3, 2 on 3) (change ED_visits==FALSE/TRUE)
paired_models <- df %>%
  filter(
    informed == TRUE,
    immunity_linking == FALSE,
    ED_visits == TRUE,
  ) %>%
  group_by(reference_date, model) %>%
  summarise(
    log_gm_WIS = mean(log(relative_WIS_nodrift), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = model,
    values_from = log_gm_WIS
  ) %>%
  drop_na()

# compute differences
paired_models <- paired_models %>%
  mutate(
    diff_1v2 = `SIR-1S` - `SIR-2S`,
    diff_1v3 = `SIR-1S` - `SIR-3S`,
    diff_2v3 = `SIR-2S` - `SIR-3S`
  )

# perform paired bootstrap
B <- 5000
boot_contrasts <- replicate(
  B,
  {
    idx <- sample(seq_len(nrow(paired_models)), replace = TRUE)
    colMeans(paired_models[idx, c("diff_1v2", "diff_1v3", "diff_2v3")])
  }
)
boot_contrasts <- t(boot_contrasts)

# transform to regular domain
summarise_contrast <- function(x) {
  tibble(
    log_ratio = mean(x),
    lower_log = quantile(x, 0.025),
    upper_log = quantile(x, 0.975),
    percentage_difference = (exp(mean(x))-1)*100,
    lower = (exp(quantile(x, 0.025))-1)*100,
    upper = (exp(quantile(x, 0.975))-1)*100
  )
}
results <- bind_rows(
  summarise_contrast(boot_contrasts[, "diff_1v2"]) %>% mutate(contrast = "SIR-1S / SIR-2S"),
  summarise_contrast(boot_contrasts[, "diff_1v3"]) %>% mutate(contrast = "SIR-1S / SIR-3S"),
  summarise_contrast(boot_contrasts[, "diff_2v3"]) %>% mutate(contrast = "SIR-2S / SIR-3S")
)

# compute p values
pvals <- tibble(
  contrast = c("SIR-1S / SIR-2S", "SIR-1S / SIR-3S", "SIR-2S / SIR-3S"),
  p_value = c(
    mean(boot_contrasts[, "diff_1v2"] <= 0),
    mean(boot_contrasts[, "diff_1v3"] <= 0),
    mean(boot_contrasts[, "diff_2v3"] <= 0)
  )
)
left_join(results, pvals, by = "contrast")


########################################
## Prepare the data for an LMER model ##
########################################

# model log transform of relative WIS
df$log_relative_WIS_drift <- log(df$relative_WIS_drift)
df$log_relative_WIS_nodrift <- log(df$relative_WIS_nodrift)

# attach months
df$month <- factor(
  format(df$reference_date, "%m"),
  levels = sprintf("%02d", 1:12),
  labels = month.abb
)

# factor relevant columns
df <- df %>%
  mutate(
    informed = factor(informed),
    immunity_linking = factor(immunity_linking),
    ED_visits = factor(ED_visits),
    model = factor(model),
    month = factor(month),
    season = factor(season)
  ) %>%
  droplevels()

##################################
## Build an additive LMER model ##
##################################

# Keep only informed
df_filtered <- df %>% filter(informed == TRUE)

# build linear mixed effects model
m_additive <- lmer(
  log_relative_WIS_nodrift ~ 
    model +
    immunity_linking + 
    ED_visits + 
    (1 | reference_date) + 
  data = df_filtered,
  REML = TRUE
)

# Conditional and marginal R2
r2_nakagawa(m_additive)

# print coefficients (of m1)
summary(m_additive)

# plot residual (lacks structure/correlation?)
plot(fitted(m_additive), resid(m_additive))

# verify normality of residuals
qqnorm(resid(m_additive))
qqline(resid(m_additive))

# do the random effects satisfy the assumption of normality?
re <- ranef(m_additive)$reference_date
qqnorm(re[, 1], main = "QQ plot of reference_date random intercepts")
qqline(re[, 1])


#####################
## Build the model ##
#####################

# build linear mixed effects model
m_full <- lmer(
  log_relative_WIS_nodrift ~ 
    informed + 
    model +
    immunity_linking + 
    ED_visits + 
    informed * model + 
    informed * immunity_linking + 
    informed * ED_visits +    
    model * ED_visits +
    model * immunity_linking + 
    (1 | reference_date) +
    (1 | season),
  data = df,
  REML = TRUE
)

# stepwise selection
model_selection <- step(m_full, keep=c('informed', 'model', 'immunity_linking', 'ED_visits'))
m1 <- get_model(model_selection)

# print coefficients (of m1)
summary(m1)



# Conditional and marginal R2
r2_nakagawa(m1)

####################
## Export results ##
####################

# compute marginal effects over informed=TRUE (+ averages out random effects)
emm <- emmeans(m1, ~ model + ED_visits + immunity_linking, at = list(informed = "TRUE"), pbkrtest.limit = 15924, lmerTest.limit = 15924)

# transform estimated effects in original domain
emm_log <- summary(emm)
emm_log$WIS <- exp(emm_log$emmean)
emm_log$lower <- exp(emm_log$lower.CL)
emm_log$upper <- exp(emm_log$upper.CL)

# basic point + errorbar plot
p <- ggplot(emm_log, aes(x = model, y = WIS, color = ED_visits)) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), 
                width = 0.2, 
                position = position_dodge(width = 0.5)) +
  facet_wrap(~ immunity_linking, labeller = labeller(immunity_linking = c("FALSE" = "(a) No immunity chaining", "TRUE" = "(b) Immunity chaining"))) +
  theme_minimal(base_size = 14) +
  labs(
    y = "Rel. WIS (GRW)",
    x = "Strain representation",
    color = "ED visits",
  ) +
  scale_color_brewer(palette = "Set1")
ggsave(
  filename = file.path(script_dir, "WIS_comparison_GRW.pdf"),
  plot = p,
  width = 8.3,
  height = 11.7/4,
  units = "in"
)

# tabulate the results
emm_log_out <- emm_log %>%
  select(
    model,
    immunity_linking,
    ED_visits,
    WIS,
    lower,
    upper
  ) %>%
  mutate(
    across(c(WIS, lower, upper), ~ round(.x, 2))
  )
emm_log_out <- emm_log_out %>%
  mutate(
    model = factor(model, levels = c("SIR-1S", "SIR-2S", "SIR-3S")),
    immunity_linking = factor(immunity_linking, levels = c("FALSE", "TRUE")),
    ED_visits = factor(ED_visits, levels = c("FALSE", "TRUE"))
  ) %>%
  arrange(model, immunity_linking, ED_visits)
write.csv(
  emm_log_out,
  file = file.path(script_dir, "predited_relative_WIS_GRW.csv"),
  row.names = FALSE
)

# compute overall difference for immunity chaining under informed==TRUE
emm <- emmeans(m1, ~ immunity_linking, at=list(informed = "TRUE"), pbkrtest.limit = 15924, lmerTest.limit = 15924)
contr <- contrast(emm, "pairwise")
summary(contr, infer = TRUE, type = "response") %>%
  transform(
    ratio = exp(estimate),
    lower = exp(lower.CL),
    upper = exp(upper.CL)
  )

# compute difference for ED_visits under informed==TRUE + immunity_linking==FALSE
emm <- emmeans(m1, ~ ED_visits, at=list(informed = "TRUE", immunity_linking="FALSE"), pbkrtest.limit = 15924, lmerTest.limit = 15924)
contr <- contrast(emm, "pairwise")
summary(contr, infer = TRUE, type = "response") %>%
  transform(
    ratio = exp(estimate),
    lower = exp(lower.CL),
    upper = exp(upper.CL)
  )

# compute difference for ED_visits per model under informed==TRUE + immunity_linking==FALSE
emm <- emmeans(m1, ~ ED_visits, at=list(informed = "TRUE", immunity_linking="FALSE"), by='model', pbkrtest.limit = 15924, lmerTest.limit = 15924)
contr <- contrast(emm, "pairwise")
summary(contr, infer = TRUE, type = "response") %>%
  transform(
    ratio = exp(estimate),
    lower = exp(lower.CL),
    upper = exp(upper.CL)
  )

