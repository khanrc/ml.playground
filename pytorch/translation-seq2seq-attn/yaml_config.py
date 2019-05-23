import yaml


def search_key(dic, key):
    """ search left-key from dict recursively """
    ret = []
    if key in dic and not isinstance(dic[key], dict):
        return [dic]
    else:
        for k, v in dic.items():
            if isinstance(v, dict):
                ret += search_key(v, key)

    return ret


class YAMLConfig:
    """ YAML configuration module

    params:
        path: yaml config file path
        argv: optional options & values arguments for modifying config in command line
    """
    def __init__(self, path, argv=None):
        if not path.endswith('.yaml'):
            path += '.yaml'

        self.path = path
        # dictionary
        self._cfg = yaml.load(open(path), Loader=yaml.Loader)
        if argv:
            self.parse_update(argv)

    def str(self):
        # string w/o comments
        return yaml.dump(self._cfg, sort_keys=False).strip()

    def markdown(self):
        # markdown
        return self.str().replace(' ', '&nbsp;').replace('\n', '  \n')

    def update(self, opts, value):
        """ update self._cfg with original type
        opts: hierarchical option list. e.g.) ["model", "n_layers"]
        value
        """
        dic = self._cfg
        # search corresponding parent dic
        if len(opts) == 1 and opts[0] not in dic:
            # leaf-key only case: e.g) ("batch_size", 128)
            # search leafs by key
            key = opts[0]
            all_parents = search_key(dic, key)
            assert len(all_parents) == 1, "# of corresponding leafs should be 1, but got {}". \
                format(len(all_parents))
            dic = all_parents[0]
        else:
            # key-path case: e.g) ("model.batch_size", 128)
            for opt in opts[:-1]:
                dic = dic[opt]
            key = opts[-1]
            assert key in dic, "{} is not the element of {}".format(key, self.path)

        type_cls = type(dic[key])
        dic[key] = type_cls(value)

    def parse_update(self, argv):
        """ parse argv & update self._cfg
        argv structure: [option1, value1, option2, value2, ...]
        """
        N = len(argv)
        assert N % 2 == 0

        for i in range(0, N, 2):
            opts = argv[i]
            value = argv[i+1]
            assert opts.startswith('--')
            opts = opts[2:].split('.')
            self.update(opts, value)

    def __repr__(self):
        return self.str()

    def __getitem__(self, key):
        return self._cfg[key]


if __name__ == "__main__":
    import sys
    argv = sys.argv
    config = YAMLConfig(argv[1])
    print(config.str())
    print("---")
    argv = argv[3:]
    print("argv:", argv)
    config.parse_update(argv)
    print(config.str())
