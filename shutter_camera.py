import numpy as np
import nidaqmx
import nidaqmx.stream_writers
import nidaqmx.stream_readers
from scipy.signal import sawtooth, square, triang
import matplotlib.pyplot as plt
from datetime import datetime




class single_DAQ_shutter():

    def __init__(self, device_name='Dev1',  
                       digital_channels=['do0'],\
                       tasks_names=['digital_task']):

        """ Constructor:
                - device_name = obvious
                - digital_channels = list with digital channels' names
                -'do0' for shutter trigger
        """
        self.device_name = device_name
        self.digital_channels = digital_channels
        self.tasks_names = tasks_names
        
        # fill the channels
        if(digital_channels!=0):
            self.digital_task = nidaqmx.Task(tasks_names[0])
            for i in range (len(self.digital_channels)):
                self.digital_task.do_channels.add_do_chan(
                                                      self.device_name
                                                      + '/port0/line31')
#-------------------------------------------------------------------------------                                                      
    def single_illumination(self, t_tot = 40, t_on = .005):
        assert(self.digital_channels == ['do0']), \
                "\n Select only channels \do0\' for this modality"
        
        """ Function to drive the shutter with single illumination. 
            t_tot= total time wave 
            duty= t_on/t_tot ----> t_on_shutter=t_tot*duty
            num_pulses=1
        """
    
               
        num_samples =  10**5
        num_pulses=1
        self.t_on=t_on
        self.duty = self.t_on/t_tot
        #time variable 
        self.t = np.linspace(0, t_tot, num_samples, dtype=np.float16)
        
        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger[:int(self.duty*len(self.t))] += 1
        self.trigger *= (2**32 - 1)
        
        #samples in one sec Changed here !!!!!
        samples_per_second = int(len(self.t))/t_tot
        # print('t_on',self.t_on, 'duty',self.duty, 't_tot', t_tot)
        # samples in one sec
        # one_second = int(len(self.t))
        
        # Define the tasks to send the digiatl  signal
        self.digital_task.timing.cfg_samp_clk_timing(rate=samples_per_second,
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))
        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        digital_writer.write_many_sample_port_uint32(self.trigger)
        
        # plt.figure('One pulse')
        # plt.plot(self.t, self.trigger, 'bo-')
        # plt.grid()
        # plt.show()
        
        t_2P_single = str(datetime.time(datetime.now()))
        
        
        #start the task and when it is done stops it, close it and reinitialize
        self.digital_task.start()
        self.digital_task.wait_until_done(float(t_tot+.05))
        self.digital_task.stop()
        self.digital_task.close()
        self.digital_task = nidaqmx.Task(self.tasks_names[0])
        for i in range (len(self.digital_channels)):
            self.digital_task.do_channels.add_do_chan(
                                                    self.device_name
                                                  + '/port0/line31')
   
        return  t_2P_single

#-------------------------------------------------------------------------------          
    
    def stop(self):
        """ Stop the task.
            Close it (forget eerything).
            Then open again.
        """
        self.digital_task.stop()
        self.digital_task.close()
 
        self.digital_task = nidaqmx.Task()
        if(self.digital_channels!=0):
            self.digital_task = nidaqmx.Task(self.tasks_names[0])
            for i in range (len(self.digital_channels)):
                self.digital_task.do_channels.add_do_chan(
                                                      self.device_name
                                                      + '/port0/line31')
#-------------------------------------------------------------------------------    
    def close(self):
        """ Completely free the task.
            Close everything.
        """
        self.digital_task.stop()
        self.digital_task.close()