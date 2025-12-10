class FirstConvUnit():
    def __init__(self,unit_id,in_size,in_channel,out_channel):
        self.unit_id=unit_id
        self.in_size=in_size
        self.in_channel=in_channel
        self.out_channel=out_channel

    def __str__(self):
        dict={"in_size":self.in_size,"in_channel":self.in_channel,"out_channel":self.out_channel}
        return "fisrt："+str(dict)