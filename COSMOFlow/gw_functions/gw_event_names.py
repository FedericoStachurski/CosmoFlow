
class GW_events(object):
    def __init__(self):
        self.events_o1 = [ 'GW150914_095045','GW151226_033853'] # O1
        self.events_o2_hlv = ['GW170809_082821', 'GW170814_103043', 'GW170818_022509'] #O2 HLV
        self.events_o2_hl = ['GW170104_101158', 'GW170608_020116', 'GW170823_131358'] #O2 HL
        self.events_o3_hlv = ['GW190408_181802', 'GW190412_053044', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428',
                          'GW190517_055101', 'GW190519_153544', 'GW190521_030229', 'GW190602_175927', 'GW190701_203306',
                          'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190828_063405', 'GW190828_065509',
                          'GW190915_235702', 'GW190924_021846', 'GW200129_065458', 'GW200202_154313', 'GW200224_222234',
                          'GW200311_115853'] #O3
        self.events_o3_hl = ['GW190521_074359', 'GW190706_222641', 'GW190707_093326', 'GW191109_010717', 'GW191129_134029',
                          'GW191204_171526','GW191222_033537', 'GW200225_060421']
        self.events_o3_hv = ['GW191216_213338']
        self.events_o3_lv = ['GW190630_185205', 'GW190708_232457', 'GW190910_112807', 'GW200112_155838' ]
        
    def get_event(self, run, detectors):
        if run == 'O1':
            return self.events_o1
        elif run == 'O2':
            if detectors == 'HLV':
                return self.events_o2_hlv 
            elif detectors == 'HL':
                return self.events_o2_hl
            else: raise ValueError('detectors not found')
        elif run == 'O3':
            if detectors == 'HLV':
                return self.events_o3_hlv
            elif detectors == 'HL':
                return self.events_o3_hl
            elif detectors == 'HV':
                return self.events_o3_hv
            elif detectors == 'LV':
                return self.events_o3_lv
            else: raise ValueError('detectors not found')