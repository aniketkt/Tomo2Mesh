#!/bin/bash

python plot_eaton_graphs.py "1" "1" "4" "1"
python plot_eaton_graphs.py "2" "1" "5" "2"
python plot_eaton_graphs.py "3" "1" "6" "3"
python plot_eaton_graphs.py "4" "1" "12" "45"
python plot_eaton_graphs.py "5" "1" "12" "45"
python plot_eaton_graphs.py "6" "1" "12" "67"
python plot_eaton_graphs.py "7" "1" "12" "67"
python plot_eaton_graphs.py "8" "1" "12" "89"
python plot_eaton_graphs.py "9" "1" "12" "89"
python plot_eaton_graphs.py "10" "1" "12" "1011"
python plot_eaton_graphs.py "11" "1" "12" "1011"
python plot_eaton_graphs.py "12" "1" "3" "12"

# for val in {1..12}; do
    # if [[$val -eq 1]]; then
    #     start_layer = 1
    #     end_layer = 4
    #     sample_tag = "1"
    # elif [[$val -eq 2 || val -eq 3]]; then
    #     start_layer = 1
    #     end_layer = 5
    #     sample_tag = val
    # elif [[$val -eq 4 || $val -eq 6 || $val -eq 8 || $val -eq 10]]; then
    #     start_layer = 1
    #     end_layer = 4
    #     if [[$val -eq 4]]; then
    #         sample_tag = "45"
    #     elif [[$val -eq 6]]; then
    #         sample_tag = "67"
    #     elif [[$val -eq 8]]; then
    #         sample_tag = "89"
    #     elif [[$val -eq 10]]; then
    #         sample_tag = "1011"
    #     fi
    # elif [[$val -eq 5 || $val -eq 7 || $val -eq 9 || $val -eq 11]]; then
    #     start_layer = 7
    #     end_layer = 10
    #     if [[$val -eq 5]]; then
    #         sample_tag = "45"
    #     elif [[$val -eq 7]]; then
    #         sample_tag = "67"
    #     elif [[$val -eq 9]]; then
    #         sample_tag = "89"
    #     elif [[$val -eq 11]]; then
    #         sample_tag = "1011"
    #     fi
    # elif [[$val -eq 12]]; then
    #     start_layer = 1
    #     end_layer = 3
    #     sample_tag = "12"
    # fi

    # python plot_eaton_graphs.py "$val" "$start_layer" "$end_layer" "$sample_tag"
#     echo $val
# done

