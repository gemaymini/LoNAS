class ResNeXtUnit():
    def __init__(self,unit_id,block_amount,in_channel,out_channel,cardinalitys,group_widths,hasSENets,stride):
        """Unit的初始化

        Args:
            unit_id (str): unit的编号
            block_amount (int): unit包含的Block数量
            in_channel (int): 第一个Block的输入特征通道
            out_channel (int): 最后一个Block的输出特征通道
            cardinalitys (list[int]): cardinality列表
            group_widths (list[int]): group_width列表
            hasSENets (list[bool]): hasSENet列表
            stride (int): 卷积层的步长
        """
        self.unit_id=unit_id #单元的编号
        self.block_amount=block_amount #单元中resnext_block的数量
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.cardinalitys=cardinalitys # 每个Block中的cardinality
        self.group_widths=group_widths # 每个Block中的group_width
        self.hasSENets=hasSENets
        self.stride=stride
    
    def __str__(self):
        dict={"block_amount":self.block_amount,"in_channel":self.in_channel,"out_channel":self.out_channel,"cardinalitys":self.cardinalitys,"group_widths":self.group_widths,"hasSENets":self.hasSENets,"stride":self.stride}
        return "resx："+str(dict)
        