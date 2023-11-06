from library.neural_network import AlexNet, LeNet

alexNet = AlexNet()
alexNet.build()

leNet = LeNet()
leNet.build(255, 255, 3, 10)