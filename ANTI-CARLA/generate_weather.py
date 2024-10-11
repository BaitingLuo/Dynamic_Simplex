import csv
stats = []
for i in range(0,101,10):
    for j in range(0,101,10):
        for k in range(0,101,10):
            temp = [i,j,k,15]
            stats.append(temp)

with open('training_weather.csv', 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in stats:
        writer.writerow(row)