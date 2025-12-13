library(tidyverse)
library(lubridate)

# path relative to repo root
path <- "bitcoining/figs/pf-3/data/daily_hl_series.csv"

df <- read_csv(path, show_col_types = FALSE) |>
  mutate(eval_date = ymd(eval_date))

# ensure all dates appear; missing dates will show as gaps (no line)
date_seq <- seq(min(df$eval_date, na.rm = TRUE), max(df$eval_date, na.rm = TRUE), by = "1 day")
df <- df |>
  complete(eval_date = date_seq)

 # Optional: clip extreme values for readability (e.g., 1st/99th pct)
 qlo <- quantile(unlist(select(df, Q1:hl_ret)), 0.01, na.rm = TRUE)
 qhi <- quantile(unlist(select(df, Q1:hl_ret)), 0.99, na.rm = TRUE)
 df  <- df |>
   mutate(across(Q1:hl_ret, ~pmin(pmax(., qlo), qhi)))

df_long <- df |>
  pivot_longer(cols = Q1:Q3, names_to = "bucket", values_to = "ret")

p <- ggplot() +
  geom_line(data = df_long, aes(x = eval_date, y = ret, color = bucket), size = 0.8) +
  geom_line(data = df, aes(x = eval_date, y = hl_ret), color = "black", linetype = "dashed", size = 0.8) +
  labs(
    x = NULL, y = "Return",
    color = "Bucket",
    title = "Daily high-minus-low series (Q1–Q3 vs HL spread)",
    subtitle = "Dashed black = HL return; colors = individual quartile series"
  ) +
  scale_color_manual(values = c("Q1" = "#4c72b0", "Q2" = "#55a868", "Q3" = "#c44e52")) +
  scale_x_date(date_breaks = "1 day", date_labels = "%Y-%m-%d", expand = expansion(mult = c(0.01, 0.01))) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
  )

print(p)
# ggsave("figs/pf-3/figures/daily_hl_series.png", p, width = 9, height = 5, dpi = 300)

# -------------------------------------------------------------------
# Bar chart: mean return per bucket
# -------------------------------------------------------------------
bucket_stats <- read_csv("bitcoining/figs/pf-3/tables/bucket_return_stats.csv", show_col_types = FALSE) |>
  mutate(bucket = factor(bucket, levels = bucket))

bucket_colors <- c(
  "1" = "#C2DDB2",
  "2" = "#A5CD8E",
  "3" = "#78B455"
)

p_bucket <- ggplot(bucket_stats, aes(x = bucket, y = mean, fill = bucket)) +
  geom_col(color = "black", width = 0.6) +
  labs(
    x = "Bucket",
    y = "Mean return"
  ) +
  scale_fill_manual(values = bucket_colors) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

print(p_bucket)
# ggsave("figs/pf-3/figures/bucket_mean_returns.png", p_bucket, width = 6, height = 4, dpi = 300)
