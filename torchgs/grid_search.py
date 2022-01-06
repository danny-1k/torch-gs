from .trainer import Trainer


class GridSearch:
    def __init__(self, param_space, trainer):
        """
        Grid search class for finding the optimal 
        set of hyperparameters

        Args:
            param_space (dict): Hyperparameters to search
            net (torch.nn.Module): Network to optimize
        """        
        self.param_space = param_space
        self.search_space = {space:self.gen_combinations(self.param_space[space]) for space in self.param_space}
        self.trainer = trainer

    def gen_combinations(self,param_space):
        """
        Function to generate all possible combinations of 
        parameters in a param_space
        """

        params = []

        if len(param_space)>1:

            for idx in range(len(param_space)-1):

                if idx == 0:
                    current_key = list(param_space.keys())[idx]
                    next_key = list(param_space.keys())[idx+1]
                    current = param_space[current_key]
                    next = param_space[next_key]

                    for param_current in current:
                        for param_next in next:
                            params.append(
                                {current_key: param_current, next_key: param_next})

                else:
                    next_key = list(param_space.keys())[idx+1]
                    next = param_space[next_key]
                    next_params = []

                    for idx, param in enumerate(params):
                        for param_next in next:
                            orig_param = param.copy()
                            orig_param[next_key] = param_next

                            next_params.append(orig_param)

                    params = next_params
        else:
            for param in param_space:
                for item in param_space[param]:
                    params.append({param:item})
            

        return params
