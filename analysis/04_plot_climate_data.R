# Visualize climate data
library(tidyverse)


# read in the data
df <- readRDS("data/climate_data_majority_vote.rds") |>
  ungroup() |>
  mutate(
    filename = basename(filename)
  )

# Split out row and columns from the filename
# as this refers to the location of the data within
# the table and hence which variables are considered
# (given a fixed format used in this data set)
df <- df |>
  mutate(
    folder = str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,3],
    image = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,4]),
    col = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,5]),
    row = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,6]),
  )

# check how many sites there are
nr_sites <- length(unique(df$filename))

# find folder with the most data
# and only retain this one
folder <- df |>
  group_by(folder) |>
  summarize(
    n = n()
  ) |>
  filter(
    n == max(n)
  )

df <- df |>
  filter(
    folder == !!folder$folder,
    col == 1
  ) |>
  mutate(
    x = paste(image, row, sep = "-")
  )

p <- ggplot(df) +
  geom_point(
    aes(
      x,
      value
    )
  ) +
  ylim(
    c(0,40)
  )

print(p)
