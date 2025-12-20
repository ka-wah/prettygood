library(ggplot2)
library(dplyr) 

## --- user settings ---------------------------------------------------------
# root directory where the results live (use / instead of \ on Windows)
base_dir <- "C:/Users/kawah/prettygood/cleanrepo/results/dh_ret/all-all"

# mapping: x-axis label -> folder name
info_sets <- data.frame(
  label  = c("I",   "I+B",   "I+B+M",   "I+B+M+C",             "I+B+M+C+T"),
  folder = c("I",   "IB",   "IBM",   "IBMCINTERACTIONS", "IBMCTINTERACTIONS"),
  stringsAsFactors = FALSE
)

target_model <- "N-En"
metrics_file <- "metrics_weighted.csv"

# bar colours
bar_colors <- c(
  I    = "#EFFBF4",
  'I+B'  = "#CFF2DF",
  'I+B+M'  = "#9DE5BE",
  'I+B+M+C' = "#6ED89E",
  'I+B+M+C+T'  = "#4ED088"
)

## --- load R2_OS and CW_t for target model in each information set ---------
labels <- character()
r2_os_w  <- numeric()
cw_t_w   <- numeric()
cols    <- character()

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
  r2_os_w  <- c(r2_os_w, row[["R2_OS_w_XS"]])
  cw_t_w   <- c(cw_t_w, row[["CW_t_w_XS"]])
  cols    <- c(cols, if (!is.na(bar_colors[lab])) bar_colors[lab] else "#999999")
}

## --- map CW t-stats to significance stars ---------------------------------
cw_stars <- function(t){
  p <- 1 - pnorm(t)  
  dplyr::case_when(
    p <= 0.001 ~ "***",  
    p <= 0.01  ~ "**",   
    p <= 0.05  ~ "*",    
    TRUE       ~ ""
  )
}

stars <- sapply(cw_t_w, cw_stars)

## --- assemble data frame for plotting -------------------------------------
plot_df <- data.frame(
  label = factor(labels, levels = labels),
  R2_OS = r2_os_w,
  CW_t  = cw_t_w,
  stars = stars,
  col   = cols,
  stringsAsFactors = FALSE
)

# Recalculate y-limits to include a small space below 0
# This is essentially what your original code did when R2_OS > 0
ymin_raw <- min(plot_df$R2_OS, 0)
ymin <- ymin_raw * 1.10 
ymax <- max(plot_df$R2_OS, 0) * 1.25

# Define a data frame for stars placed below the axis
stars_df <- plot_df %>%
  mutate(y_pos = ymin_raw * 0.10) # Position the stars just below the axis line (near 0)


## --- Define the desired plotting style (theme_paper) ---
theme_paper <- theme_classic(base_size = 12) +
  theme(
    legend.position    = "none", 
    panel.grid.major.y = element_line(colour = "grey80", linetype = "dashed"),
    panel.grid.minor   = element_blank(),
    axis.title.x       = element_blank(), 
    plot.margin        = margin(t = 5, r = 5, b = 10, l = 5),
    axis.line          = element_line(colour = "black"),
    axis.ticks         = element_line(colour = "black")
  )

## --- make bar plot --------------------------------------------------------
p <- ggplot(plot_df, aes(x = label, y = R2_OS, fill = label)) +
  geom_col(color = "black", width = 0.6) +
  scale_fill_manual(values = setNames(plot_df$col, plot_df$label)) +
  # Use linetype "longdash" for the zero line
  geom_hline(yintercept = 0, linetype = "longdash", linewidth = 0.5) +
  
  # New geom_text to place stars below the axis
  geom_text(
    data = stars_df, # Use the stars_df with the explicit y_pos
    aes(x = label, y = y_pos, label = stars),
    vjust = 1.2, # Adjust vertical position to sit below the axis
    size = 4,
    inherit.aes = FALSE # Ensure it uses only the data specified here
  ) + 
  
  labs(x = NULL, y = expression(R[OSXS]^2)) +
  
  # Keep coord_cartesian with the automatically calculated ymin/ymax to retain the space
  coord_cartesian(ylim = c(ymin, ymax)) +
  
  # Remove the scale_y_continuous line to restore default padding/space
  
  # Apply the custom theme
  theme_paper 

print(p)