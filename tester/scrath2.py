gain = [1.9080611737315443, 1.4653359793993523, 0.8293368967235644, 0.5204364617067254, 0.3845513725245052, 0.3605326073452423, 0.29845577448804034, 0.4175450978047853]
wo_gain = [1.9035692789694298, 1.4256312722256674, 1.2856009249632545, 0.6860931887585093, 0.2810121678811563, 0.2852267331393728, 0.273466465142253, 0.2635620227101601]
final = 0.2

sum_num = 0
for i in gain[:5]:
    sum_num += final - i

sum_den = 0
for i in wo_gain[:5]:
    sum_den += final - i

print(sum_num)
print(sum_den)
print(1 - (sum_num/sum_den))
