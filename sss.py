# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:26:16 2024

@author: Administrator
"""
import random

# Given previous winning numbers
previous_red_numbers = [2, 9, 12, 19, 21, 31]
previous_blue_number = 4

# Generate the next most likely red numbers (6 numbers from 1 to 33 excluding previous winning numbers)
possible_red_numbers = [num for num in range(1, 34) if num not in previous_red_numbers]
next_red_numbers = random.sample(possible_red_numbers, 6)

# Generate the next most likely blue number (1 number from 1 to 16 excluding previous winning number)
possible_blue_numbers = [num for num in range(1, 17) if num != previous_blue_number]
next_blue_number = random.choice(possible_blue_numbers)

print("Next most likely red numbers:", next_red_numbers)
print("Next most likely blue number:", next_blue_number)