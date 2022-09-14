# data to process
library(tidyverse)

df <- readr::read_csv("data-raw/cnn_column_stats.csv")

# total coverage of min / max temperature + rainfall
# minimum set of data

values <- df %>%
  filter(name == "minimum_temperature" |
         name == "maximum_temperature" |
         name == "rainfall_mm"
         ) %>%
  group_by(index) %>%
  summarize(total_rows = sum(row_count),
            n_records = length(unique(img))) %>%
  arrange(desc(total_rows), desc(n_records))
message("---------------------------")
message("training data:")
message("total sheets")

values %>%
  filter(n_records / 12 >= 6) %>%
  summarize(sum(n_records)) %>%
  message()

message("total values")
values %>%
  filter(n_records / 12 >= 6) %>%
  summarize(sum(total_rows)) %>%
  message()

message("validation data:")

message("total values")
values %>%
  filter(n_records / 12 < 6) %>%
  summarize(sum(n_records)) %>%
  message()

message("total sheets")
values %>%
  filter(n_records / 12 < 6) %>%
  summarize(sum(total_rows)) %>%
  message()

message("---------------------------")
message("overall total values to process for format 1")
values %>%
  summarize(sum(total_rows)) %>%
  message()

sel_1 <- values %>%
  filter(n_records / 12 >= 6)

sel_2 <- values %>%
  filter(n_records / 12 < 6)

# list all format 1 files
img_files <- list.files("/backup/cobecore/sorted_images/format_1/",
                        "*",
                        recursive = TRUE,
                        full.names = TRUE)

# sort batch 1 files
batch_1 <- do.call("c",apply(sel_1,1, function(s){
  img_files[grep(s['index'],basename(img_files))]
}))

# write batch 1 to file
write.table(batch_1,
            "data-raw/format_1/format_1_batch_1.csv",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE,
            sep = ",")

# sort batch 1 files
batch_2 <- do.call("c",apply(sel_2,1, function(s){
  img_files[grep(s['index'],basename(img_files))]
}))

# write batch 1 to file
write.table(batch_2,
            "data-raw/format_1/format_1_batch_2.csv",
            quote = FALSE,
            row.names = FALSE,
            col.names = FALSE,
            sep = ",")



