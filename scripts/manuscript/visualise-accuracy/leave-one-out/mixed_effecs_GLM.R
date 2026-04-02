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

# Factor formatting
df_boot <- df %>%
  mutate(
    informed = factor(informed),
    model = factor(model, levels = c("SIR-1S", "SIR-2S", "SIR-3S")),
    immunity_linking = factor(immunity_linking, levels = c(FALSE, TRUE)),
    ED_visits = factor(ED_visits, levels = c(FALSE, TRUE))
  )

# paired bootstrap function
boot_geom_mean_ratio <- function(df_group, baseline_col, B = 20) {
  refdates <- unique(df_group$reference_date)
  n_dates <- length(refdates)
  
  boot_stats <- replicate(B, {
    # sample reference dates WITH replacement
    sampled_dates <- tibble(
      reference_date = sample(refdates, size = n_dates, replace = TRUE)
    )
    # join to preserve multiplicity
    sample_df <- sampled_dates %>%
      left_join(df_group, by = "reference_date")
    # compute geometric mean ratio
    log_mean_model <- mean(log(sample_df$WIS), na.rm = TRUE)
    log_mean_baseline <- mean(log(sample_df[[baseline_col]]), na.rm = TRUE)
    exp(log_mean_model - log_mean_baseline)
  })
  
  # point estimate on full data
  log_mean_model <- mean(log(df_group$WIS), na.rm = TRUE)
  log_mean_baseline <- mean(log(df_group[[baseline_col]]), na.rm = TRUE)
  
  tibble(
    geom_mean_rel_WIS = exp(log_mean_model - log_mean_baseline),
    lower_95 = quantile(boot_stats, 0.025, na.rm = TRUE),
    upper_95 = quantile(boot_stats, 0.975, na.rm = TRUE)
  )
}

# Apply bootstrap (nonstationary)
boot_tbl_nonstat <- df_boot %>%
  group_by(informed, model, immunity_linking, ED_visits) %>%
  summarise(
    stats = list(boot_geom_mean_ratio(cur_data(), "WIS_GRW_nonstationary")),
    .groups = "drop"
  ) %>%
  unnest(stats) %>%
  mutate(baseline = "nonstationary")

# Apply bootstrap (stationary)
boot_tbl_stat <- df_boot %>%
  group_by(informed, model, immunity_linking, ED_visits) %>%
  summarise(
    stats = list(boot_geom_mean_ratio(cur_data(), "WIS_GRW_stationary")),
    .groups = "drop"
  ) %>%
  unnest(stats) %>%
  mutate(baseline = "stationary")

# Combine results
boot_tbl_out <- bind_rows(boot_tbl_nonstat, boot_tbl_stat) %>%
  mutate(
    geom_mean_rel_WIS = round(geom_mean_rel_WIS, 2),
    lower_95 = round(lower_95, 2),
    upper_95 = round(upper_95, 2)
  ) %>%
  arrange(baseline, informed, model, immunity_linking, ED_visits)

# Save output
write.csv(
  boot_tbl_out,
  file = file.path(script_dir, "bootstrapped_relative_WIS.csv"),
  row.names = FALSE
)

# Visualise output
selected_baseline <- "nonstationary"  # or "stationary"
p <- ggplot(
  boot_tbl_out %>% 
    filter(informed == TRUE, baseline == selected_baseline),
  aes(x = model, y = geom_mean_rel_WIS, color = ED_visits)
) +
  geom_point(position = position_dodge(width = 0.5), size = 3) +
  geom_errorbar(
    aes(ymin = lower_95, ymax = upper_95),
    width = 0.2,
    position = position_dodge(width = 0.5)
  ) +
  facet_wrap(
    ~ immunity_linking,
    labeller = labeller(
      immunity_linking = c(
        "FALSE" = "(a) No immunity chaining",
        "TRUE"  = "(b) Immunity chaining"
      )
    )
  ) +
  theme_minimal(base_size = 14) +
  labs(
    y = paste0(
      "Rel. WIS (", 
      ifelse(selected_baseline == "nonstationary", 
             "nsGRW", 
             "sGRW"),
      ")"
    ),
    x = "No. Strains Represented",
    color = "ED visits"
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
  filename = file.path(
    script_dir, 
    paste0("WIS_comparison_", selected_baseline, ".pdf")
  ),
  plot = p,
  width = 8.3,
  height = 11.7 / 4,
  units = "in"
)

#############################################
## Prove informed==FALSE << informed==TRUE ##
#############################################

# compute difference in geom_mean between informed FALSE/TRUE per reference date
# baseline model cancels out
paired_df <- df %>%
  group_by(reference_date, informed) %>%
  summarise(
    log_gm_WIS = mean(log(WIS), na.rm = TRUE),
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
  diff_percent = (exp(mean(boot_log_diff)) - 1) * 100,
  lower = (exp(ci[1]) - 1) * 100,
  upper = (exp(ci[2]) - 1) * 100
)
result

# no need to prove statistical significance here

############################################################
## Prove immunity linking doesn't work for informed==TRUE ##
############################################################

# I don't think we have to prove that it doesn't work for the groups ED_visits and model
# Folks will be able to see on the figure immunity_linking never achieves lower WIS
# It only barely achieves similar WIS scores to having no immunity linking

# compute paired difference between immunity_linking TRUE/FALSE per reference_date
paired_immune <- df %>%
  filter(
    informed == TRUE,
    immunity_linking %in% c(TRUE, FALSE)
  ) %>%
  group_by(reference_date, immunity_linking) %>%
  summarise(
    log_gm_WIS = mean(log(WIS), na.rm = TRUE),  # CDC-standard
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

# perform paired bootstrap by reference_date
B <- 10000
boot_log_diff <- replicate(
  B,
  {
    idx <- sample(seq_len(nrow(paired_immune)), replace = TRUE)
    mean(paired_immune$log_diff[idx])
  }
)

# report results (percent change relative to FALSE)
ci <- quantile(boot_log_diff, c(0.025, 0.975))
result <- tibble(
  log_ratio = mean(boot_log_diff),
  lower_log = ci[1],
  upper_log = ci[2],
  diff_percent = (exp(mean(boot_log_diff)) - 1) * 100,
  lower = (exp(ci[1]) - 1) * 100,
  upper = (exp(ci[2]) - 1) * 100
)
result

# one-sided p-value: prob that immunity linking is better (log_diff < 0)
p_value <- mean(boot_log_diff <= 0)
p_value

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
    log_gm_WIS = mean(log(WIS), na.rm = TRUE),  # CDC-standard
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
p_value

###############################################################################
## Find out if ED_visits works for informed==TRUE and immunity_linking==TRUE ##
###############################################################################

# compute paired difference between groups (add model == 'SIR-1S' to test model-by-model ED_visits works)
paired_ED_visits <- df %>%
  filter(
    informed == TRUE,
    immunity_linking == TRUE,
  ) %>%
  group_by(reference_date, ED_visits) %>%
  summarise(
    log_gm_WIS = mean(log(WIS), na.rm = TRUE),
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
p_value

##################################################################################################
## Prove SIR-1S < SIR-2S < SIR-3S for informed==TRUE, immunity_linking==FALSE, ED_visits==FALSE ##
## Prove SIR-1S < SIR-2S = SIR-3S for informed==TRUE, immunity_linking==FALSE, ED_visits==TRUE  ##
##################################################################################################

# compute paired difference between models (1 on 2, 1 on 3, 2 on 3) (change ED_visits==FALSE/TRUE)
paired_models <- df %>%
  filter(
    informed == TRUE,
    immunity_linking == FALSE,
    ED_visits == FALSE,
  ) %>%
  group_by(reference_date, model) %>%
  summarise(
    log_gm_WIS = mean(log(WIS), na.rm = TRUE),
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
B <- 10000
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
# df$log_relative_WIS_stationary <- log(df$relative_WIS_stationary)
# df$log_relative_WIS_nonstationary <- log(df$relative_WIS_nonstationary)
df$log_WIS <- log(df$WIS)
  
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
  log_WIS ~ 
    model +
    immunity_linking + 
    ED_visits + 
    (1 | reference_date), 
  data = df_filtered,
  REML = TRUE
)

# Conditional and marginal R2
r2_nakagawa(m_additive)

# print coefficients (of m1)
summary(m_additive)

# plot residual (lacks structure/correlation?)
plot(fitted(m_additive), resid(m_additive))

# open figure
pdf(file.path(script_dir,"qqplots_lmer_diagnostics.pdf"), width = 10, height = 5)
par(mfrow = c(1, 2))

# set scaling factors
title_cex <- 2      # main titles
axis_cex  <- 1.5    # tick labels
label_cex <- 1.5    # axis labels (if you add them)

qqnorm(resid(m_additive),
       main = "Residuals",
       cex.main = title_cex,
       cex.lab  = label_cex,
       cex.axis = axis_cex)
qqline(resid(m_additive))

re <- ranef(m_additive)$reference_date
qqnorm(re[, 1],
       main = "Random Intercepts",
       cex.main = title_cex,
       cex.lab  = label_cex,
       cex.axis = axis_cex)
qqline(re[, 1])

dev.off()

