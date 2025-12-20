# ==============================================================================
# MASTER SCRIPT: THESIS PLOTS (FINAL WITH OOS LINE)
# ==============================================================================

if (!require("tidyverse")) install.packages("tidyverse")
if (!require("arrow")) install.packages("arrow")
if (!require("scales")) install.packages("scales")

library(tidyverse)
library(arrow)
library(scales)

# 1. SET LOCALE & THEME
tryCatch(Sys.setlocale("LC_TIME", "C"), error = function(e) message("Could not set locale to C"))

theme_paper <- theme_classic(base_size = 12) +
  theme(
    legend.position      = "none", 
    panel.grid.major.y   = element_line(colour = "grey80", linetype = "dashed"),
    panel.grid.minor     = element_blank(),
    axis.title.x         = element_blank(),
    plot.margin          = margin(t = 5, r = 15, b = 10, l = 5),
    axis.line            = element_line(colour = "black"),
    axis.ticks           = element_line(colour = "black")
  )

# 2. LOAD DATA
cat("Loading parquet file...\n")
df <- read_parquet("spread0.8/dhinput.parquet") 

cat("Processing dates...\n")
clean_df <- df %>%
  mutate(day = as.Date(date)) %>%
  filter(!is.na(day))

dir.create("results/dh_ret/figures/datasec/", recursive = TRUE, showWarnings = FALSE)

# --- DEFINING THE OOS START DATE ---
oos_start_date <- as.Date("2023-03-01")

# ------------------------------------------------------------------------------
# FIGURE 2: DATA INTEGRITY (Histogram with OOS Line)
# ------------------------------------------------------------------------------
cat("Generating Figure 2...\n")

fig2_data <- clean_df %>%
  filter(dte >= 1, dte <= 31, opt_rel_spread_raw <= 0.8) %>%
  count(day, name = "n_contracts")

p2 <- ggplot(fig2_data, aes(x = day, y = n_contracts)) +
  geom_col(fill = "#404040", width = 1) +
  
  # Vertical Line for OOS Split
  geom_vline(xintercept = oos_start_date, linetype = "dashed", color = "#6A8474", linewidth = 0.5) +
  
  # Optional: Add "OOS" label
  annotate("text", x = oos_start_date + 10, y = max(fig2_data$n_contracts) * 0.95, 
           label = "OOS", hjust = 0, size = 3, fontface = "italic") +
  
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y", expand = c(0, 0)) +
  scale_y_continuous(labels = comma, expand = c(0, 0)) +
  labs(y = "Number of Contracts") +
  theme_paper

ggsave("results/dh_ret/figures/datasec/figure2_data_integrity_final.png", plot = p2, width = 8, height = 5, dpi = 300)


# ------------------------------------------------------------------------------
# FIGURE 3: CUMULATIVE PNL (With OOS Line)
# ------------------------------------------------------------------------------
cat("Generating Figure 3...\n")

fig3_data <- clean_df %>%
  group_by(day) %>%
  summarise(daily_pnl = -sum(dh_pnl, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(day) %>%
  mutate(
    cumulative_pnl = cumsum(daily_pnl),
    high_water_mark = cummax(cumulative_pnl)
  )

p3 <- ggplot(fig3_data, aes(x = day)) +
  geom_hline(yintercept = 0, linetype = "longdash", linewidth = 0.5, color = "black") +
  
  # Vertical Line for OOS Split
  geom_vline(xintercept = oos_start_date, linetype = "dashed", color = "#6A8474", linewidth = 0.5) +
  annotate("text", x = oos_start_date + 10, y = 750000 * 0.95, 
           label = "OOS", hjust = 0, size = 3, fontface = "italic") +
  geom_ribbon(aes(ymin = cumulative_pnl, ymax = high_water_mark), fill = "#cc0000", alpha = 0.15) +
  geom_line(aes(y = cumulative_pnl), color = "#000000", linewidth = 0.8) +
  
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y", expand = c(0, 0)) +
  scale_y_continuous(labels = dollar_format(prefix = "$", scale = 1e-3, suffix = "k")) +
  labs(y = "Cumulative PnL") +
  theme_paper

ggsave("results/dh_ret/figures/datasec/figure3_cumulative_pnl_final.png", plot = p3, width = 8, height = 5, dpi = 300)

cat("Done. Figures updated with OOS line (March 1, 2023).\n")