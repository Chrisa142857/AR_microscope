import os

class cfg:

    def __init__(self, microType, slideType):

        self.microType = microType
        self.slideType = slideType
        self.scale = 1
        if microType == '3d':
            self.scale = 2.411

            if slideType == 'sfy1_pos':
                self.label_dir = 'G:/wei/original_data/Shengfuyou_1th/Labelfiles/csv_P/'
                self.slide_dir = 'G:/wei/original_data/Shengfuyou_1th/Positive/'
            elif slideType == 'goldtest':
                self.label_dir = 'G:/wei/original_data/Shengfuyou_1th/Labelfiles/csv_GoldTest/'
                self.slide_dir = 'G:/wei/original_data/Shengfuyou_1th/GoldTest/'
            elif slideType == 'sfy2_pos':
                self.label_dir = 'G:/wei/original_data/Shengfuyou_2th/Labelfiles/csv_P/'
                self.slide_dir = 'G:/wei/original_data/Shengfuyou_2th/Positive/'
            elif slideType == 'sfy1_neg':
                self.label_dir = 'neg'
                self.slide_dir = 'G:/wei/original_data/Shengfuyou_1th/Negative/'
            elif slideType == 'sfy2_neg':
                self.label_dir = 'neg'
                self.slide_dir = 'G:/wei/original_data/Shengfuyou_2th/Negative/'
            if self.label_dir != 'neg':
                for r, csvs, f in os.walk(self.label_dir):
                    break
                for r, d, svss in os.walk(self.slide_dir):
                    break
                self.dataIds = []
                for x in csvs:
                    if x+'.mrxs' in svss:
                        self.dataIds.append(x)
                    else:
                        print(x,' Unfound')
            else:
                for r, d, svss in os.walk(self.slide_dir):
                    break
                self.dataIds = []
                for x in svss:
                    self.dataIds.append(x[:-5])

        elif microType == 'our':
            self.scale = 2

            for r, csvs, f in os.walk(self.label_dir):
                break
            for r, d, svss in os.walk(self.slide_dir):
                break
            self.dataIds = []
            for x in csvs:
                if x+'.svs' in svss:
                    self.dataIds.append(x)
                else:
                    print(x,' Unfound')

        elif microType == 'szsq':
            self.scale = 3.2501

            if slideType == 'tj3_pos':
                self.label_dir = 'G:/wei/original_data/Tongji_3th/Labelfiles/'
                self.slide_dir = 'G:/wei/original_data/Tongji_3th/positive/'
            elif slideType == 'tj4_pos':
                self.label_dir = 'G:/wei/original_data/Tongji_4th/Labelfiles/'
                self.slide_dir = 'G:/wei/original_data/Tongji_4th/positive/'
            elif slideType == 'tj3_neg':
                self.label_dir = 'neg'
                self.slide_dir = 'G:/wei/original_data/Tongji_3th/negative/'
            elif slideType == 'tj4_neg':
                self.label_dir = 'neg'
                self.slide_dir = 'G:/wei/original_data/Tongji_4th/negative/'
            if self.label_dir != 'neg':
                for r, d, xmls in os.walk(self.label_dir):
                    break
                for r, d, svss in os.walk(self.slide_dir):
                    break
                self.dataIds = []
                for x in xmls:
                    if x[:-4]+'.sdpc' in svss:
                        self.dataIds.append(x[:-4])
                    else:
                        print(x[:-4],' Unfound')
            else:
                for r, d, svss in os.walk(self.slide_dir):
                    break
                self.dataIds = []
                for x in svss:
                    self.dataIds.append(x[:-5])

                
        for r, d, f in os.walk(self.slide_dir):
            break
        if len(f) == 0:
            self.slide_tail = ''
        else:
            filename = f[0]
            self.slide_tail = filename[filename.rfind('.'):]

        for r, d, f in os.walk(self.label_dir):
            break
        if len(f) == 0:
            self.tail = '.csv'
        else:
            self.tail = '.xml'
        
