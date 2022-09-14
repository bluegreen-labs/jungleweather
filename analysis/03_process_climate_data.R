# Jungle Weather statistics

# Process headers of format 1 data as annotated by the
# CitSci volunteers. This reads in the annotations CSV
# splits out the meta-dat workflow and selects a majority
# vote result while retaining uncertain data for
# re-interpretation

# load the tidyverse for data wrangling
library(tidyverse)

# read in the small demo data using read.table(), this might
# change for final processing as read.table() is slow for larger
# files
df <- read.table(
  "~/Downloads/jungle-weather-classifications.csv",
  header = TRUE,
  sep = ",",
  stringsAsFactors = FALSE)

# filter unnecessary data which slow down processing
df <- df |>
  select(
    -user_name,
    -user_ip,
    -workflow_name
  )

# Filter out the meta-data workflow (9713)
# (grepping on meta-data for the workflow name is not correct
# as the tests data will then be included)
df <- df |>
  filter(
    workflow_id == 9712
  )

# Get month and year values from the embedded
# JSON in the CSV (read data frame)
df <- df |>
  mutate(
    filename = as.vector(unlist(
      lapply(
      subject_data,
      function(x){
        data <- as.data.frame(t(unlist(jsonlite::fromJSON(x))))
        data <- data |> 
          select(contains("Filename")) |>
          unname()
        return(data)
        }))),
    value = as.vector(unlist(
      lapply(annotations, function(x){
        data <- jsonlite::fromJSON(x)
        data <- as.numeric(data$value[1])
        return(data)
      })))
  )

# Flag unclear values, those which were not easily read
# and are certainly due for reprocessing
df <- df |>
  mutate(
    unclear = ifelse(is.na(value), TRUE, FALSE)
  )

# Split out row and columns from the filename
# as this refers to the location of the data within
# the table and hence which variables are considered
# (given a fixed format used in this data set)
df <- df |>
  mutate(
    folder = str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,3],
    image = as.numeric(str_split(tools::file_path_sans_ext(df$filename),"_", simplify = TRUE)[,4]),
    col = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,5]),
    row = as.numeric(str_split(tools::file_path_sans_ext(filename),"_", simplify = TRUE)[,6]),
  )

# Group data by subject_id for majority vote analysis
# and summaries, I report the number of classifications
# on record, those that were marked unclear (missing),
# and the number of unique values (the higher this number)
# the higher the variability
majority_vote <- df |>
  group_by(subject_ids, filename) |>
  summarize(
    nr_classifications = n(),
    value_sd = sd(value, na.rm = TRUE),
    nr_values = length(unique(value)),
    value = as.numeric(names(which.max(table(value)))),
    nr_unclear = length(which(unclear))
  )

# save results to the data directory
write.table(
  majority_vote,
  "data/climate_data_majority_vote.csv",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE,
  sep = ","
)

saveRDS(
  majority_vote,
  "data/climate_data_majority_vote.rds",
  compress = "xz"
  )