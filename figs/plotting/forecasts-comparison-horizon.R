# --- packages ---
library(tidyverse)

# --- user inputs ---
base_dir   <- "C:/Users/kawah/prettygood/cleanrepo/results/dh_ret"
spec_dir   <- "IBMCTINTERACTIONS" 

group_map <- c("all-all" = "All", "call-all" = "Call", "put-all" = "Put")
mat_map   <- c("dte_1_7" = "Ultra-Short", "dte_8_31" = "Short")

# --- helpers ---
cw_stars <- function(t){
  p <- 1 - pnorm(t)
  case_when(
    p <= 0.001 ~ "***", 
    p <= 0.01  ~ "**",  
    p <= 0.05  ~ "*",   
    TRUE       ~ ""
  )
}

pretty_model <- function(x){
  key <- tolower(gsub("#\\d+$", "", x))
  case_when(
    key %in% c("l-en", "l_en", "len", "linear-ensemble") ~ "L-En",
    key %in% c("n-en", "n_en", "nen", "nl-en", "nl_en", "nlen", "nonlinear-ensemble") ~ "N-En",
    TRUE ~ NA_character_
  )
}

read_single_file <- function(g_folder, m_folder) {
  f <- file.path(base_dir, g_folder, spec_dir, m_folder, "metrics.csv")
  if(!file.exists(f)) return(NULL)
  
  read_csv(f, show_col_types = FALSE) |>
    mutate(
      OptionType = group_map[[g_folder]],
      Maturity   = mat_map[[m_folder]]
    )
}

# --- load data ---
df_list <- list()
i <- 1
for (g in names(group_map)) {
  for (m in names(mat_map)) {
    df_list[[i]] <- read_single_file(g, m)
    i <- i + 1
  }
}
raw <- bind_rows(df_list)

# --- clean & organize ---
plot_data <- raw |>
  mutate(model_clean = pretty_model(model)) |>
  filter(!is.na(model_clean)) |>
  select(-any_of("R2_OS")) |> 
  select(-any_of("CW_t")) |> 
  rename(R2_OS = R2_OS_XS, CW_t = CW_t_XS) |>
  
  # Deduplicate
  group_by(Maturity, OptionType, model_clean) |>
  slice_max(R2_OS, n = 1, with_ties = FALSE) |>
  ungroup() |>
  
  # Ordering
  mutate(
    Maturity   = factor(Maturity, levels = c("Ultra-Short", "Short")),
    model      = factor(model_clean, levels = c("L-En", "N-En")),
    OptionType = factor(OptionType, levels = c("Call", "Put", "All"))
  )

# --- Create separate dataframe for stars ---
# We select only the "All" rows to get the significance, 
# but we will NOT map OptionType in the plot layer, effectively centering it.
stars_df <- plot_data |>
  filter(OptionType == "All") |>
  mutate(
    raw_stars = cw_stars(CW_t),
    y_star    = -0.01
  )

# --- plotting ---
fill_vals <- c("Call" = "#EC9AC0", "Put" = "#6A8474", "All" = "#505679")

theme_paper <- theme_classic(base_size = 12) +
  theme(
    legend.position = "top",
    panel.grid.major.y = element_line(colour = "grey85", linetype = "dashed"),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank(),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text = element_text(face = "bold", size = 11, margin = margin(b=10)),
    plot.margin = margin(t = 5, r = 10, b = 10, l = 5)
  )

p <- ggplot(plot_data, aes(x = model, y = R2_OS, fill = OptionType)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.4) +
  
  facet_wrap(~Maturity, scales = "fixed") + 
  
  # --- Add Stars Centered ---
  # We use the separate stars_df. 
  # We do NOT map 'fill' or 'group' inside aes(), and we do NOT use position_dodge.
  # This places the text at the center of the x-tick.
  geom_text(
    data = stars_df,
    aes(x = model, y = y_star, label = raw_stars),
    inherit.aes = FALSE, # Prevent inheriting the 'fill' from the main plot
    vjust = 0.8, 
    size = 4
  ) +
  
  scale_fill_manual(values = fill_vals, name = NULL) +
  labs(y = expression(R[RAWXS]^2)) +
  # Ensure y-axis can fit the stars below 0
  coord_cartesian(ylim = c(-0.02, 0.164*1.1)) + 
  theme_paper

# --- save ---
out_dir <- file.path(base_dir, "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(out_dir, "R2_OS_Maturity_CenteredStars.png"), 
       p, width = 8, height = 4.5, dpi = 300)

print(p)