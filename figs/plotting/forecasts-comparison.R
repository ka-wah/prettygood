# --- packages ---
library(tidyverse)

# --- user inputs ---
#base_dir  <- "C:/Users/kawah/bitcoining/big_results"
base_dir = "C:/Users/kawah/Documents/bitcoining/results/y_price"
spec_dir  <- "IBMCTINTERACTIONS"   # change if you switch spec
group_dirs <- c(All = "all-all", Call = "call-all", Put = "put-all")


# --- helpers ---
pretty_model <- function(x){
  # strip run suffix like "#1"
  key <- tolower(gsub("#\\d+$", "", x))
  # map to paper labels
  case_when(
    key == "rf"                     ~ "RF",
    key == "lgbm_gbdt"              ~ "GBR",
    key == "lgbm_dart"              ~ "DART",
    key == "elasticnet"             ~ "ENet",
    key == "pcr"                    ~ "PCR",
    key == "pls"                    ~ "PLS",
    key == "ridge"                  ~ "Ridge",
    key == "lasso"                  ~ "Lasso",
    key == "ffn"                    ~ "FFN",
    key %in% c("l-en","l_en","len") ~ "L-En",
    key %in% c("nl-en","nl_en","nlen") ~ "N-En",
    TRUE                            ~ x
  )
}

cw_stars <- function(t){
  p <- 1 - pnorm(t)  # one-sided upper tail
  dplyr::case_when(
    p <= 0.001 ~ "***",  # 0.1%
    p <= 0.01  ~ "**",   # 1%
    p <= 0.05  ~ "*",    # 5%
    TRUE      ~ ""
  )
}


read_one_group <- function(base_dir, group_dir, spec_dir){
  f <- file.path(base_dir, group_dir, spec_dir, "metrics_returns.csv")
  read_csv(f, show_col_types = FALSE) |>
    mutate(group = names(group_dirs)[group_dirs == group_dir])
}

# --- load + tidy ---
raw <- map_dfr(unname(group_dirs), ~read_one_group(base_dir, .x, spec_dir)) |>
  rename(
    model_raw = model,
    R2_OS     = R2_OS,
    R2_OS_XS  = R2_OS_XS,
    CW_t      = CW_t,
    CW_t_XS   = CW_t_XS          # must exist in metrics_target.csv
  ) |>
  mutate(model = pretty_model(model_raw))

raw <- raw |>
  filter(model != "NN-En")

if (!"CW_t_XS" %in% names(raw)) {
  stop("Column 'CW_t_XS' not found in metrics_cw.csv; check the file.")
}

# collapse multiple runs per model/group to the single variant
# keeping the row with the highest R2_OS_XS
metrics <- raw |>
  group_by(group, model) |>
  slice_max(order_by = R2_OS_XS, with_ties = FALSE) |>
  ungroup() |>
  mutate(
    stars_OS    = cw_stars(CW_t),      # stars based on CW_t      (R2_OS)
    stars_OS_XS = cw_stars(CW_t_XS)    # stars based on CW_t_XS   (R2_OS_XS)
  )

# quick check: see what stars you actually get for All group
print(
  metrics |>
    filter(group == "All") |>
    select(model, R2_OS, CW_t, stars_OS, R2_OS_XS, CW_t_XS, stars_OS_XS)
)

# choose model order like the paper when available
paper_order <- c("OLS", "Ridge","Lasso","ENet","PCR","PLS","L-En","GBR","DART","RF","FFN","N-En")
present <- intersect(paper_order, unique(metrics$model))
others  <- setdiff(unique(metrics$model), paper_order)

metrics <- metrics |>
  mutate(
    model = factor(model, levels = c(present, sort(others))),
    group = factor(group, levels = c("Call","Put","All")) # legend order: Call, Put, All
  )

# --- plotting style ---
fill_vals <- c("Call" = "#D2FFFB", "Put" = "#CBCBCB", "All" = "#966A46")

theme_paper <- theme_classic(base_size = 12) +
  theme(
    legend.position    = "top",
    panel.grid.major.y = element_line(colour = "grey80", linetype = "dashed"),
    panel.grid.minor   = element_blank(),
    axis.title.x       = element_blank(),
    plot.margin        = margin(t = 5, r = 5, b = 10, l = 5)
  )

# ======================================================
# Figure 1: R2_OS with stars (All group, based on CW_t)
# ======================================================

# stars placed just above the All bar for each model
stars_os_df <- metrics |>
  filter(group == "All", R2_OS > 0) |>
  mutate(
    y = -0.01      # place stars just below the x-axis
  )

p1 <- ggplot(metrics, aes(x = model, y = R2_OS, fill = group)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "longdash", linewidth = 0.5) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  geom_text(
    data = stars_os_df,
    aes(x = model, y = y, label = stars_OS, group = group),
    position = position_dodge(width = 0.75),
    vjust = 0,
    size = 4,
    inherit.aes = FALSE
  ) +
  labs(y = expression(R[OS]^2)) +
  theme_paper +
  coord_cartesian(ylim = c(-0.02, 0.12))   # <-- choose ymin, ymax here


# ============================================================
# Figure 2: R2_OS_XS with stars (All group, based on CW_t_XS)
# ============================================================

stars_xs_df <- metrics |>
  filter(group == "All", R2_OS_XS > 0) |>
  mutate(
    y = -0.01   # place stars just below the x-axis
  )

p2 <- ggplot(metrics, aes(x = model, y = R2_OS_XS, fill = group)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "longdash", linewidth = 0.5) +
  scale_fill_manual(values = fill_vals, name = NULL) +
  geom_text(
    data = stars_xs_df,
    aes(x = model, y = y, label = stars_OS_XS, group = group),
    position = position_dodge(width = 0.75),
    vjust = 0,
    size = 4,
    inherit.aes = FALSE
  ) +
  labs(y = expression(R[OSXS]^2)) +
  theme_paper +
  coord_cartesian(ylim = c(-0.02, 0.075))   # <-- and here


# --- save (optional) ---
out_dir <- file.path(base_dir, "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(out_dir, "R2_OS_by_model_with_stars.png"),
       p1, width = 8, height = 4, dpi = 300)
ggsave(file.path(out_dir, "R2_OS_XS_by_model_with_stars.png"),
       p2, width = 8, height = 4, dpi = 300)

# Print to the plot window
p1
p2
