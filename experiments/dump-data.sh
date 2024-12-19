#!/bin/bash

# 检查是否提供了文件名
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

filename=$1

# 检查文件是否存在
if [ ! -f "$filename" ]; then
  echo "Error: File '$filename' not found."
  exit 1
fi

# 提取"Time elapsed in looking up"字段的浮点值
values=($(grep -oP 'Time elapsed in inserting \d+ items: \K[0-9]+\.[0-9]+' "$filename"))

# 初始化变量
count=0
sum=0

# 遍历提取的值，按每5个分组计算平均值
for value in "${values[@]}"; do
  sum=$(echo "$sum + $value" | bc)
  count=$((count + 1))

  # 每5个值输出平均值
  if (( count == 5 )); then
    average=$(echo "scale=2; $sum / $count" | bc)
    echo "Average of group: $average"
    count=0
    sum=0
  fi
done

# 处理剩余未满5个的值
if (( count > 0 )); then
  average=$(echo "scale=2; $sum / $count" | bc)
  echo "Average of remaining group: $average"
fi
