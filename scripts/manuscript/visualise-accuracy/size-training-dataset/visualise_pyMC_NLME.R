library(readr)
library(dplyr)
library(tidyr)
library(glue)
library(ggplot2)

baseline <- 'drift'
if (baseline == 'nodrift') {baseline_label <- 'sGRW'} else {baseline_label <- 'nsGRW'}

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
data <- read_csv(file.path(script_dir, glue("pyMC_output/2plot_{baseline}.csv")))

data <- data %>%
  mutate(
    ED     = factor(ED, levels = c(0, 1), labels = c("No ED visits", "ED visits")),
  )


###############
## Visualize ##
###############

p <- ggplot(data) +
  geom_point(
    aes(x = training_horizon, y = empirical_gm, color = model, shape = model),
    size = 1.5,
  ) +
  geom_line(
    aes(x = training_horizon, y = modeled_gm, color = model),
    linewidth = 0.5,
  ) +
  facet_grid(ED ~ season) +
  scale_y_continuous(
    name = glue("Rel. WIS ({baseline_label})"),
    limits = c(0.45, 1.35)
  ) +
  scale_x_continuous(
    name = "Number of training seasons"
  ) +
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey95"),
    panel.grid.minor = element_blank()
  )
ggsave(
  filename = file.path(script_dir, glue("pyMC_output/training_model_{baseline_label}.pdf")),
  plot = p,
  width = 8.3,
  height = 11.7/3,
  units = "in"
)
                 