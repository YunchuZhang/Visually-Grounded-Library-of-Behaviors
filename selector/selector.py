
class Selector:
    def __init__(self, policy_name):
        self.selector_name = selector_name


    def compare_objects(self, env, obj):
        # output should be a dictionary
        # 'avg_reward': float32
        # 'success_rate': float32
        # using env to 
        raise NotImplementedError("Must be implemented in subclass.")



