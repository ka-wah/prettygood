# 1. Install and load necessary packages
if(!requireNamespace("tidyverse", quietly = TRUE)) {
  install.packages("tidyverse")
}
if(!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}
library(tidyverse)
library(arrow)
install.packages("paletteer")
library(paletteer)

# Use in a ggplot2 chart:
scale_colour_paletteer_d("lisa::AndyWarhol_3")
scale_fill_paletteer_d("lisa::AndyWarhol_3")

# --- 2. Load the Parquet Data ---
# IMPORTANT: Replace "your_data_file.parquet" with the actual name of your file.
data <- read_parquet("C:/Users/kawah/Documents/bitcoining/outputs-vega/features_and_dh_returns.parquet")

# --- 3. Data Cleaning and Preparation ---
data_clean <- data %>%
  # Ensure the key column is numeric and filter out NA/infinite values
  mutate(moneyness_std = as.numeric(moneyness_std)) %>%
  filter(!is.na(moneyness_std), is.finite(moneyness_std))

# --- 4. Calculate Mean and Standard Deviation ---
moneyness_stats <- data_clean %>%
  summarise(
    Mean_Moneyness_Std = mean(moneyness_std, na.rm = TRUE),
    StdDev_Moneyness_Std = sd(moneyness_std, na.rm = TRUE)
  )

# Print the statistics
cat("--- Standardized Moneyness Statistics ---\n")
print(moneyness_stats)
cat("-----------------------------------------\n")


# --- 5. Create the Distribution Plot (Histogram with Density Overlay) ---
moneyness_plot <- ggplot(data_clean, aes(x = moneyness_std)) +
  # Histogram Layer
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 50,
    # fill = "#FF1493",
    # color = "white",
    alpha = 0.8
  ) +
  # Density Curve Layer
  geom_density(
    #color = "#FF7F0E",
    linewidth = 1.2
  ) +
  # Add labels and titles
  labs(
    title = "Distribution of Standardized Moneyness (moneyness_std)",
    x = "Standardized Moneyness",
    y = "Density"
  ) +
  # Apply a clean theme
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10))
  )
moneyness_plot + scale_colour_paletteer_d("lisa::AndyWarhol_3")
#scale_fill_paletteer_d("lisa::AndyWarhol_3")

# --- 6. Display and Save the Plot ---
print(moneyness_plot)

# Save the plot to a PNG file
# ggsave("bitcoining/figures/moneyness_std_distribution.png", plot = moneyness_plot, width = 9, height = 6)