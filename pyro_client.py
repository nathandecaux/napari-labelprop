import Pyro5.api

greeting_maker = Pyro5.api.Proxy("PYRONAME:example.greeting")    # use name server object lookup uri shortcut
print(greeting_maker.get_shape())