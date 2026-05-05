import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from torch import nn as nn
    import marimo as mo
    from matplotlib import pyplot as plt

    return mo, nn, plt, torch


@app.cell
def _(mo, nn, plt, torch):
    # We are going to do linear regression; y=2x+1
    # Define the data
    X = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
    y = torch.tensor([[3.0],[5.0],[5.0],[9.0]])

    # Model
    # nn.Linear(1,1) means 1 input number (x) and 1 output number (y)
    model = nn.Linear(in_features=1,out_features=1)

    # Optimizer and Loss Function
    # MSE
    criterion = nn.MSELoss()
    # Stochastic Gradient Descent
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    # using Adam's optimizer instead
    optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

    # tracking the loss
    loss_history = []

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        y_predicted = model(X) # make prediction

        loss = criterion(y_predicted,y) # compute loss

        optimizer.zero_grad()
        loss.backward() # I am guessing this is backprop
        optimizer.step() # Apply SGD

        loss_history.append(loss.item())

    fig,ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("How the Error Dropped Over Time"
                )
    # Result
    learned_weight = model.weight.item()
    learned_bias = model.bias.item()

    mo.vstack([mo.as_html(fig), mo.md(f"""### Training is done!
    Model learned **y = {learned_weight:.2f}x+{learned_bias:.2f}**""")])
    return ax, criterion, fig, loss, model


@app.cell
def _(mo):
    # Sliders
    n_neurons_slider = mo.ui.slider(start=2,stop=128,step=2,value=32,label="Number of Neurons")
    lr_slider = mo.ui.slider(start=0.001,stop=0.1,step=0.001,value=0.01,label="Learning Rate")
    return lr_slider, n_neurons_slider


@app.cell
def _(ax, criterion, lr_slider, model, n_neurons_slider, nn, plt, torch):
    # nonlinear model: try to learn y=x^2

    X2 = torch.linspace(-2,2,50).view(-1,1)
    y2 = X2**2

    model_quadratic = nn.Sequential(
        # Layer 1
        nn.Linear(1,n_neurons_slider.value),
        nn.ReLU(), # activation function

        nn.Linear(n_neurons_slider.value,n_neurons_slider.value),
        nn.ReLU(), 

        nn.Linear(n_neurons_slider.value,1)
    )

    optimizer_quadratic = torch.optim.Adam(model.parameters(),lr=lr_slider.value)
    criterion_quadratic = nn.MSELoss()

    # training
    for epoch_quadratic in range(5000):
        y_pred_quadratic = model(X2)
        loss_quadratic = criterion(y_pred_quadratic,y2)

        optimizer_quadratic.zero_grad()
        loss_quadratic.backward()
        optimizer_quadratic.step()

    fig2,ax2 = plt.subplots()
    ax2.scatter(X2.numpy(),y2.numpy(),label="Real Data",color="gray",alpha=0.5)
    ax2.plot(X2.numpy(),model_quadratic(X2).detach().numpy(),label="Model Prediction",color="red",linewidth=3)
    ax.set_title(f"Model with {n_neurons_slider.value} Neurons")
    ax2.legend()
    return


@app.cell
def _(fig, loss, lr_slider, mo, n_neurons_slider):
    mo.vstack([
        mo.md("## The Neural Network Lab"),
        mo.hstack([n_neurons_slider,lr_slider],justify="start"),
        mo.as_html(fig),
        mo.md(f"**Final Loss:** {loss.item():.6f}")
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
