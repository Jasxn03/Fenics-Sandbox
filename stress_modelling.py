import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch
import gpytorch

df = pd.read_csv("simulation_results/stress_full.csv")
# x = df[["height", "length", "velocity"]].values
x = df[["height", "length", "velocity", "x", "y"]].values
y = df["stress"].values
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)
y_scaled = np.log1p(y)

# region Poly model
# model = Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)), ("linreg", LinearRegression())])

# model.fit(x,y)
# print("R^2:", model.score(x,y))
# linreg = model.named_steps["linreg"]
# coef = linreg.coef_
# intercept = linreg.intercept_
# poly = model.named_steps["poly"]
# feature_names = poly.get_feature_names_out(["height", "length", "velocity"])
# for name, c in zip(feature_names,coef):
#     print(f"{c:.5e}*{name}")
# print("intercept:", intercept)
#endregion

#region GP model
# kernel = ConstantKernel(1.0,(1e-3,1e3)) * RBF(length_scale=(1,1,1))
# gp = Pipeline([("scaler", StandardScaler()), ("gp", GaussianProcessRegressor(kernel=kernel, alpha= 1e-6, normalize_y=True, n_restarts_optimizer=5))])
# gp.fit(x,y)
# print("gp trained")

# y_pred, y_std = gp.named_steps["gp"].predict(gp.named_steps["scaler"].transform([[9.0,20.0,11500]]),return_std= True)

# print("Predicted stress:", y_pred[0])
# print("Uncertainty:", y_std[0])

# y_pred_gp = gp.named_steps["gp"].predict(gp.named_steps["scaler"].transform(x))
# r2_gp = r2_score(y,y_pred_gp)
# print("R^2 gp:" ,r2_gp)
#endregion

#region Sparse GP model
# scaler_x = StandardScaler()
# x_scaled = scaler_x.fit_transform(x)
# y_scaled = (y-y.mean())/y.std()

# x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
# y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# class SparseGPModel(gpytorch.models.ApproximateGP):
#     def __init__(self,inducing_points):
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
#         variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
#         super().__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = x.shape[1]))
#     def forward(self,x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# M = 500
# inducing_points = x_tensor[torch.randperm(x_tensor.size(0))[:M]]
# model = SparseGPModel(inducing_points)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()

# model.train()
# likelihood.train()
# optimizer = torch.optim.Adam([{'params': model.parameters()},{'params': likelihood.parameters()}], lr=0.01)
# mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x_tensor.size(0))

# n_iter = 200
# for i in range(n_iter):
#     optimizer.zero_grad()
#     output = model(x_tensor)
#     loss = -mll(output,y_tensor)
#     loss.backward()
#     if i %20==0:
#         print(f"Iter{i}/{n_iter}-Loss:{loss.item():.4f}")
#     optimizer.step()

# model.eval()
# likelihood.eval()

# new_x = np.array([[9.0,20.0,11500]])
# new_x_scaled = scaler_x.transform(new_x)
# new_x_tensor = torch.tensor(new_x_scaled, dtype=torhc.float32)

# with torch.no_grad(), gpy.torch.settings.fast_pred_var():
#     pred_dist = likelihood(model(new_x_tensor))
#     mean = pred_dist.mean.itme()
#     std = pred_dist.stddev.item()

# print("Predicted stress:", mean)
# print("uncertainty:", std)
#endregion

#region RandomForest model

x_train, x_test, y_train, y_test = train_test_split(x_scaled,y_scaled,test_size=0.2, random_state= 42)
rf = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

print("R^2:", r2)
print("RMSE:", rmse)

new_point = [[9.0,20.0,11500, 25, 9.0]]
predicted_stress = rf.predict(new_point)[0]
print("Predicted stress:", predicted_stress)

import matplotlib.pyplot as plt
plt.bar(["height", "length", "velocity", "x", "y"],rf.feature_importances_)
plt.ylabel("importance")
plt.title("feature importance in rf")
plt.show()

#endregion