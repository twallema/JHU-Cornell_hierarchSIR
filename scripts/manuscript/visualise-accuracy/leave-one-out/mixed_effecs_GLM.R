library(readr)
library(lme4)
library(MMWRweek)
library(ggplot2)
library(dplyr)
library(emmeans)

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

######################
## Prepare the data ##
######################

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
    month = factor(month)
  ) %>%
  droplevels()

#####################
## Build the model ##
#####################

# build linear mixed effects model
m1 <- lmer(
  log_relative_WIS_nodrift ~ 
    informed + 
    model +
    immunity_linking + 
    ED_visits + 
    month +
    model * month + 
    model * ED_visits +
    model * immunity_linking + 
    informed * model + 
    informed * immunity_linking + 
    informed * ED_visits +
    (1 | reference_date),
  data = df,
  REML = TRUE
)

# print coefficients (of m0)
summary(m1)

# visualise fit versus residual to validated LMER noise assumption
plot(fitted(m1), resid(m1))

# compute marginal effects over informed=TRUE
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

# save as PDF with explicit size
ggsave(
  filename = file.path(script_dir, "WIS_comparison_GRW.pdf"),
  plot = p,
  width = 8.3,
  height = 11.7/4,
  units = "in"
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

