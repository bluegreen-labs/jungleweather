# Load Libraries -----

library(tools)
library(tidyverse)

# Read data -----

# read in the nilco index meta-data
df <- read.csv("./data-raw/state_archive_index/nilco_climate_station_meta_data.csv",
                 sep = ",",
                 header = TRUE,
               stringsAsFactors = FALSE)

files <- list.files("/scratch/cobecore/formatted_scans/format_1/",
           "*labels.csv",
           recursive = TRUE,
           full.names = TRUE)

# report row and columns in output file!!
labels <- do.call("rbind", 
  lapply(files, function(file){
  read.table(file, header = TRUE, stringsAsFactors = FALSE,
                       sep = ",")
  })
)

labels$col <- apply(labels, 1, function(x){
    as.numeric(strsplit(x["files"], "_")[[1]][5])
  })

labels$row <- apply(labels, 1, function(x){
  as.numeric(strsplit(file_path_sans_ext(x["files"]), "_")[[1]][6])
})

labels$index <- apply(labels, 1, function(x){
  as.numeric(strsplit(x["files"], "_")[[1]][3])
})

labels$img <- apply(labels, 1, function(x){
  as.numeric(strsplit(x["files"], "_")[[1]][4])
})

header <- c(
  "maximum_temperature",
  "minimum_temperature",
  "average_temperature",
  "temperature_amplitude",
  "dry_bulb_temperature",
  "wet_bulb_temperature",
  "relative_humidity",
  "rainfall_mm",
  "rainfall_duration",
  "rainfall_intensity",
  "evaporation",
  "soil_state",
  "thunder_distance",
  "thunder_duration",
  "thunder_intensity",
  "notes"
)

header <- cbind(header, 1:length(header))
colnames(header) <- c("name","col")

col_stats <- labels %>%
  group_by(index, img, col) %>%
  summarize(row_count = length(which(cnn_labels == "complete")))

col_stats <- merge(col_stats, header, by.x = "col")

write.table(col_stats, "data-raw/cnn_column_stats.csv",
            col.names = TRUE,
            row.names = FALSE,
            quote = FALSE,
            sep = ",")