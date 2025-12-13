library(ggplot2)

## --- user settings ---------------------------------------------------------
# root directory where the results live  (use / instead of \ on Windows)
base_dir <- "C:/Users/kawah/Documents/bitcoining/results/y_price/all-all"

# mapping: x-axis label -> folder name
info_sets <- data.frame(
  label  = c("I",   "I+B",   "I+B+M",   "I+B+M+C",             "I+B+M+C+T"),
  folder = c("I",   "IB",   "IBM",   "IBMCINTERACTIONS", "IBMCTINTERACTIONS"),
  stringsAsFactors = FALSE
)

target_model <- "N-En"
metrics_file <- "metrics_returns.csv"

# bar colours
bar_colors <- c(
  I    = "#C2DDB2",
  'I+B'   = "#A5CD8E",
  'I+B+M'  = "#78B455",
  'I+B+M+C' = "#5B8E3E",
  'I+B+M+C+T'  = "#40632C"
)

## --- load R2_OS and CW_t for target model in each information set ---------
labels <- character()
r2_os  <- numeric()
cw_t   <- numeric()
cols   <- character()

base_model <- strsplit(target_model, "#")[[1]][1]

for (i in seq_len(nrow(info_sets))) {
  lab    <- info_sets$label[i]
  folder <- info_sets$folder[i]
  path   <- file.path(base_dir, folder, metrics_file)

  if (!file.exists(path)) {
    message(sprintf("Skipping missing file: %s", path))
    next
  }

  df <- read.csv(path, stringsAsFactors = FALSE)

  # allow either exact match or numbered suffix (e.g., elasticnet#1)
  mask <- startsWith(df$model, paste0(base_model, "#"))
  row  <- df[mask, ]

  if (nrow(row) == 0) {
    row <- df[df$model == target_model, ]
  }
  if (nrow(row) == 0) {
    stop(sprintf("Model %s not found in %s", target_model, path))
  }

  row <- row[1, ]
  labels <- c(labels, lab)
  r2_os  <- c(r2_os, row[["R2_OS"]])
  cw_t   <- c(cw_t, row[["CW_t"]])
  cols   <- c(cols, if (!is.na(bar_colors[lab])) bar_colors[lab] else "#999999")
}

## --- map CW t-stats to significance stars ---------------------------------
stars_from_t <- function(t) {
  at <- abs(t)
  if (at >= 3.29) {
    "***"
  } else if (at >= 2.58) {
    "**"
  } else if (at >= 1.96) {
    "*"
  } else {
    ""
  }
}

stars <- sapply(cw_t, stars_from_t)

## --- assemble data frame for plotting -------------------------------------
plot_df <- data.frame(
  label = factor(labels, levels = labels),
  R2_OS = r2_os,
  CW_t  = cw_t,
  stars = stars,
  col   = cols,
  stringsAsFactors = FALSE
)

ymax <- max(plot_df$R2_OS, 0) * 1.25
ymin <- min(plot_df$R2_OS, 0) * 1.10
y_bracket <- ymax * 0.90

## --- make bar plot --------------------------------------------------------
p <- ggplot(plot_df, aes(x = label, y = R2_OS, fill = label)) +
  geom_col(color = "black", width = 0.6) +
  scale_fill_manual(values = setNames(plot_df$col, plot_df$label)) +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  # significance stars
  geom_text(aes(label = stars, y = R2_OS + (ymax - ymin) * 0.02),
            vjust = 0, size = 3) +
  labs(x = NULL, y = expression(R[OS]^2)) +
  coord_cartesian(ylim = c(ymin, ymax)) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

print(p)
