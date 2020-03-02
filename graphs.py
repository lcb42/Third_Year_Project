import numpy as np
import matplotlib.pyplot as plt

test_rmses = [6.997155779421233, 6.823995597600929, 6.8355326407543275, 6.863748126769217, 6.7333639987886, 6.939274307361951, 6.972099389067703, 6.633434539769307, 6.882922838795619, 6.9499404774871945, 6.85311620948543, 6.837340871556547]
train_rmses = [6.661785401601475, 6.570354736076237, 6.576632691296676, 6.590795195095073, 6.508777679123215, 6.725665381530511, 6.673946542522539, 6.340418231016112, 6.615638607960221, 6.745028973785695, 6.552833002595432, 6.608519863882665]

ns = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]

plt.plot(ns, test_rmses, "b*", label="Test RMSE")
plt.plot(ns, train_rmses, "r*", label="Train RMSE")
plt.plot(ns, test_rmses, c='b')
plt.plot(ns, train_rmses, c='r')
plt.ylabel("RMSE Score")
plt.xlabel("Number of Each Type of Estimator ")
#plt.xscale("log")
plt.legend()
plt.show()