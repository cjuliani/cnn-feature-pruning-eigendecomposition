import pruning as pr

# Get classifier and initialize
model = pr.classifier()

# Get pruning system
pr_syst = pr.pruning_system(model)

# Check initial prediction metrics
_, _ = pr_syst.mdl.predict(N=pr_syst.mdl.imgs_.shape[0], intv=10, show_avg=True, show_pgr=True)

# Check conv layers' activation space
pr_syst.check_layer(n_examples=2, layer_name='conv92', activation='Relu', index=':0', save=False, show=True)

# start pruning
layer_names = ['conv12','conv22','conv32','conv42','conv52','conv62','conv72','conv82','conv92']
pr_syst.pruning(layer_names=layer_names, epsilon=0.005, activation='Relu', index=':0')


