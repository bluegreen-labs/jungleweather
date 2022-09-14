# Format manifests for format 1

# normal manifest generation
#format_manifest(path = "/backup/cobecore/zooniverse/format_1_batch_1/")

# correct manifest after upload barfed
# use default settings
data <- read_subject_file() 

format_manifest(
  path = "/backup/cobecore/zooniverse/format_1_batch_1/",
  exclude_cell_files = data
  )