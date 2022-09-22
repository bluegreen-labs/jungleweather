# process data coverage
library(tools)
library(tidyverse)

df <- read.table("data-raw/cnn_column_stats.csv",
                 header = TRUE,
                 sep = ",",
                 stringsAsFactors = FALSE)

index <- read.csv("data-raw/state_archive_index/nilco_climate_station_meta_data.csv",
                    sep = ",",
                    header = TRUE,
                    stringsAsFactors = FALSE)

df <- merge(df, index, by.x = "index")
df <- as.tibble(df)


df_stats <- df %>% 
  group_by(name) %>%
  summarize(completed = length(which(row_count > 10)),
            total_rows = length(row_count),
            completed_perc = round((completed / total_rows)*100),
            years = total_rows / 12) %>%
  arrange(desc(completed_perc))


write.table(df_stats, "~/stats.csv",
            row.names = FALSE,
            col.names = TRUE,
            sep = ",",
            quote = FALSE)

label <- df_stats$name
label <- gsub("_"," ", label)
y <- df_stats$completed_perc

bc <- data.frame(y,label)

library(ggplot2)
library(ggthemes)

ggplot(bc) + 
  geom_bar(aes(x = as.factor(label), y = y), stat = "identity") +
  coord_flip() +
  scale_x_discrete(limits = as.factor(label)) +
  labs(x = "", y = "", title = "% Coverage") +
  theme_fivethirtyeight()
