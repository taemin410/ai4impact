

def main(args):

    # Load configurations 
    configs = load_config()

    # Preprocess the data
    data =  Data(configs)
    model = Model(configs)

    model.train()
    pickle_save()

    # Evaluation phase
    output = model.eval()

    log(output)



# visualize(output)



