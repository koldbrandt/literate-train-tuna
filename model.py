class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=8,              
                out_channels=8,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=8,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),
            nn.ReLU())

        self.fully_connected = nn.Sequential(
                nn.Linear(14*14*16, 500),
                nn.ReLU(),
                nn.Linear(500, 10),
                nn.Softmax(dim=1))
    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x