class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Action):
            return removePrivateFields(obj.__dict__)
            # return {'type':, obj[0], }
        return super(NpEncoder, self).default(obj)