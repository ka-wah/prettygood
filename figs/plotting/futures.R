# ==============================================================================
# FIGURE 1: BTC PRICE + REALIZED VOLATILITY (DUAL AXIS)
# ==============================================================================

if (!require("tidyverse")) install.packages("tidyverse")
if (!require("zoo")) install.packages("zoo") # For rolling volatility
if (!require("scales")) install.packages("scales")

library(tidyverse)
library(zoo)
library(scales)

# 1. SET LOCALE & THEME
tryCatch(Sys.setlocale("LC_TIME", "C"), error = function(e) message("Could not set locale to C"))

theme_paper <- theme_classic(base_size = 12) +
  theme(
    legend.position      = "none", 
    # Add right axis line and ticks for the secondary axis
    axis.line.y.right    = element_line(color = "#cc0000"),
    axis.ticks.y.right   = element_line(color = "#cc0000"),
    axis.text.y.right    = element_text(color = "#cc0000"),
    axis.title.y.right   = element_text(color = "#cc0000", angle = 90),
    
    panel.grid.major.y   = element_line(colour = "grey90", linetype = "dashed"), # Lighter grid
    panel.grid.minor     = element_blank(),
    axis.title.x         = element_blank(),
    plot.margin          = margin(t = 5, r = 15, b = 10, l = 5),
    axis.line            = element_line(colour = "black"),
    axis.ticks           = element_line(colour = "black")
  )

# 2. LOAD & PROCESS DATA
# Replace with your actual filename (e.g., "futures_cme-long.csv")
# Note: Based on your snippet, I'm assuming a standard CSV structure
cat("Processing futures data...\n")

# Mocking the load for the script logic (Replace read_csv with your file path)
df_fut <- read_csv("data/futures_cme-long.csv") 

# Assuming 'df_fut' is loaded with columns: date, close
# We process it:
plot_data <- df_fut %>%
  mutate(
    # Parse "Dec 29 2023" -> Date Object
    date = as.Date(date, format = "%b %d %Y"),
    price = as.numeric(close)
  ) %>%
  arrange(date) %>%
  mutate(
    # 1. Log Returns
    log_ret = log(price / lag(price)),
    
    # 2. 30-Day Realized Vol (Annualized)
    # Using 252 trading days for CME Futures standard
    rv_30 = rollapply(log_ret, width = 30, FUN = sd, fill = NA, align = "right") * sqrt(252)
  ) %>%
  filter(!is.na(rv_30)) %>%
  filter(date >= as.Date("2020-01-01")) # align with your thesis start date

# 3. CALCULATE SCALING FACTOR FOR DUAL AXIS
# ggplot requires we multiply the secondary series to match the primary range
max_price <- max(plot_data$price, na.rm = T)
max_vol   <- max(plot_data$rv_30, na.rm = T)
scale_factor <- max_price / max_vol

# 4. PLOT
oos_start_date <- as.Date("2023-03-01")

p1 <- ggplot(plot_data, aes(x = date)) +
  
  # --- OOS Vertical Line ---
  geom_vline(xintercept = oos_start_date, linetype = "dashed", color = "#6A8474", linewidth = 0.5) +
  # Optional: Add "OOS" label
  annotate("text", x = oos_start_date + 10, y = 70000 * 0.95, 
           label = "OOS", hjust = 0, size = 3, fontface = "italic") +
  # --- Volatility (Secondary Axis) ---
  # We assume Volatility is the "background" context, so we plot it first (behind)
  # or as a lighter line. 
  geom_line(aes(y = rv_30 * scale_factor), color = "#cc0000", alpha = 0.4, linewidth = 0.6) +
  
  # --- Price (Primary Axis) ---
  geom_line(aes(y = price), color = "black", linewidth = 0.8) +
  
  # --- Axes ---
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y", expand = c(0, 0)) +
  
  # Primary Y Axis (Price)
  scale_y_continuous(
    labels = dollar_format(prefix = "$", scale = 1e-3, suffix = "k"),
    expand = c(0, 0),
    limits = c(0, max_price * 1.05), # give a little headroom
    
    # Secondary Y Axis (Volatility) configuration
    sec.axis = sec_axis(~ . / scale_factor, 
                        name = "30d Realized Volatility", 
                        labels = percent_format(accuracy = 1))
  ) +
  
  # Labels
  labs(
    y = "BTC Futures Price"
  ) +
  
  theme_paper

# Save
dir.create("results/dh_ret/figures/datasec/", recursive = TRUE, showWarnings = FALSE)
ggsave("results/dh_ret/figures/datasec/figure1_regimes.png", plot = p1, width = 8, height = 5, dpi = 300)

cat("Done. Figure 1 saved.\n")