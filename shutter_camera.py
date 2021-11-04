import numpy as np
import nidaqmx
from nidaqmx.constants import  LineGrouping
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
        # samples_per_second = int(len(self.t))/t_tot
        samples_per_second = int(len(self.t))
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
    def train_illumination(self, frequency= 0.05, t_on = .005, num_pulses=10):
        
        """ Function to drive the shutter with a train of pulses. 
            f= frequency of square wave 
            duty= t_on*f ----> t_on_shutter=duty/f
            num_pulses=num repetions 
        """
        
        assert(self.digital_channels == ['do0']), \
                "\n Select only channels \do0\' for this modality"
                
        num_samples =  10**5
        
        t_tot=1/frequency
        self.t_on=t_on
        self.duty = self.t_on/t_tot
        #time variable 
        self.t = np.linspace(0, 1/frequency, int(num_samples), dtype=np.float16)
         # self.t = np.linspace(0, 1/frequency, int(num_samples/num_pulses), dtype=np.float16)
        #-------------------------------------------------
        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        #single square wave with scipy
        # self.trigger = (2**32 - 1) * \
        #          (square(2 * np.pi * self.t*frequency , self.duty)+1)/2.
        #self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger[:int(self.duty*len(self.t))] += 1
        self.trigger *= (2**32 - 1)
        #Total pulses train = repetitions of single square wave 
        self.trigger = np.tile(self.trigger, num_pulses)
        self.trigger = self.trigger.astype(np.uint32)
        #samples per second
        samples_per_second = int(len(self.t)*frequency)
        
        # samples in one sec
        # one_second = int(len(self.t)*frequency)
        
        # plot to check 
        # plt.figure('Pulses train')
        # plt.plot(np.tile(self.t, num_pulses), self.trigger, 'bo-')
        # # plt.plot(self.t, self.trigger, 'bo-')
        # plt.grid()
        # plt.show()
        
        
        # Define the tasks to send the digiatl  signal
        self.digital_task.timing.cfg_samp_clk_timing(rate=samples_per_second,
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))

        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        digital_writer.write_many_sample_port_uint32(self.trigger)
        
        self.t_2P_train = []
        t_2P_train =str(datetime.time( datetime.now()))
        self.t_2P_train.append(t_2P_train)
        
        self.digital_task.start()
        self.digital_task.wait_until_done(float(t_tot*num_pulses+.05))
        self.digital_task.stop()
        self.digital_task.close()
        self.digital_task = nidaqmx.Task(self.tasks_names[0])
        for i in range (len(self.digital_channels)):
            self.digital_task.do_channels.add_do_chan(
                                                    self.device_name
                                                  + '/port0/line31')
        
        return self.t_2P_train

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





class DAQ_shutter_im_digital():

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
            
            self.digital_task.do_channels.add_do_chan(self.device_name  + '/port0/', line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
               
#-------------------------------------------------------------------------------                                                      



#------------------------------------------------------------------------------
    def single_illumination_imaging (self, t_tot = 40, t_on = .005, frequency=0.05, n_im=10):
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
        #samples in one sec Changed here !!!!!
        samples_per_second = int(len(self.t))/t_tot

        #time variable imaging 
        self.t_im=np.linspace(0, 1/frequency, num_samples, dtype=np.float16)
        #samples in one sec imaging 
        samples_per_second = int(len(self.t))*frequency


        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger_sh = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger_sh[:int(self.duty*len(self.t))] += 1
        self.trigger_sh *= (2**32 - 1)


        #Imaging signal 
        self.trigger_im = np.zeros((int(num_samples)), dtype=np.float16)
        trigger_im = 5 * (square(2 * np.pi * self.t * frequency, duty = .1)+1)/2.
        # trigger_im[int(len(trigger_im)*3/4+1):] = 0
        self.trigger_im[:] = trigger_im[:]
        
     
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
        
        plt.figure('One pulse')
        plt.plot(self.t, self.trigger, 'bo-')
        plt.grid()
        plt.show()
        
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
    def train_illumination(self, frequency= 0.05, t_on = .005, num_pulses=10):
        
        """ Function to drive the shutter with a train of pulses. 
            f= frequency of square wave 
            duty= t_on*f ----> t_on_shutter=duty/f
            num_pulses=num repetions 
        """
        
        assert(self.digital_channels == ['do0']), \
                "\n Select only channels \do0\' for this modality"
                
        num_samples =  10**5
        
        t_tot=1/frequency
        self.t_on=t_on
        self.duty = self.t_on/t_tot
        #time variable 
        self.t = np.linspace(0, 1/frequency, int(num_samples), dtype=np.float16)
         # self.t = np.linspace(0, 1/frequency, int(num_samples/num_pulses), dtype=np.float16)
        #-------------------------------------------------
        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        #single square wave with scipy
        # self.trigger = (2**32 - 1) * \
        #          (square(2 * np.pi * self.t*frequency , self.duty)+1)/2.
        #self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger[:int(self.duty*len(self.t))] += 1
        self.trigger *= (2**32 - 1)
        #Total pulses train = repetitions of single square wave 
        self.trigger = np.tile(self.trigger, num_pulses)
        self.trigger = self.trigger.astype(np.uint32)
        #samples per second
        samples_per_second = int(len(self.t)*frequency)
        
        # samples in one sec
        # one_second = int(len(self.t)*frequency)
        
        # plot to check 
        # plt.figure('Pulses train')
        # # plt.plot(np.tile(self.t, num_pulses), self.trigger, 'bo-')
        # plt.plot(self.t, self.trigger, 'bo-')
        # plt.grid()
        # plt.show()
        
        
        # Define the tasks to send the digiatl  signal
        self.digital_task.timing.cfg_samp_clk_timing(rate=samples_per_second,
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))

        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        digital_writer.write_many_sample_port_uint32(self.trigger)
        
        self.t_2P_train = []
        t_2P_train =str(datetime.time( datetime.now()))
        self.t_2P_train.append(t_2P_train)
        
        self.digital_task.start()
        self.digital_task.wait_until_done(float(t_tot*num_pulses+.05))
        self.digital_task.stop()
        self.digital_task.close()
        self.digital_task = nidaqmx.Task(self.tasks_names[0])
        for i in range (len(self.digital_channels)):
            self.digital_task.do_channels.add_do_chan(
                                                    self.device_name
                                                  + '/port0/line31')
        
        return self.t_2P_train

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


#--------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class  DAQ_shutter_im_digital_analog(object):
    """Class to control the light sheet acquisition : plane galvo, depth galvo, camera trigger, 
        etl scan (analog channels) and shutter (digital channel). There are functions to acquire
        time lapses, time lapses +  shutter single pulse, time lapse +  shutter train pulse.  
    """

    def __init__(self, device_name='Dev1', 
                       analog_channels=['ao3'], 
                       digital_channels=['do0'],\
                       tasks_names=['digital_task', 'analog_task']):

        """ Constructor:
                - device_name = obvious;
                - analog_channels = list with analog channels' names
        
                    -'ao3' for camera trigger
                - digital_channels = list with digital channels' names
                    -'do0' for shutter trigger
        """
        self.device_name = device_name
        self.analog_channels = analog_channels
        self.digital_channels = digital_channels
        self.tasks_names = tasks_names
        
        # fill the channels
        if(digital_channels!=0):
            self.digital_task = nidaqmx.Task(tasks_names[0])
            for i in range (len(self.digital_channels)):
                self.digital_task.do_channels.add_do_chan(
                                                      self.device_name
                                                      + '/port0/line31')
        if(analog_channels!=0):  
            self.analog_task = nidaqmx.Task(tasks_names[1])
             
            for i in range (len(self.analog_channels)):   
                self.analog_task.ao_channels.add_ao_voltage_chan(
                                                    self.device_name
                                                    +'/'+self.analog_channels[i])
          
                      
#-------------------------------------------------------------------------------
    def stop(self):
        """ Stop the task.
            Close it (forget eerything).
            Then open again.
        """

        
        if(self.digital_channels!=0):
            self.digital_task.stop()
            self.digital_task.close()
            self.digital_task = nidaqmx.Task(self.tasks_names[0])
            for i in range (len(self.digital_channels)):
                self.digital_task.do_channels.add_do_chan(
                                                      self.device_name
                                                      + '/port0/line31')
                                                      
                                                      
        if(self.analog_channels!=0):
            self.analog_task.stop()
            self.analog_task.close()
            self.analog_task = nidaqmx.Task(self.tasks_names[1])

            for i in range (len(  self.analog_channels)):
            # add the active channels to the task
                self.analog_task.ao_channels.add_ao_voltage_chan(\
                                    self.device_name+'/'+self.analog_channels[i])
            # default everything to 0
        # if(len(self.analog_channels) == 2):
        #     self.analog_task.write([0, 0])
        # else:
        #     self.analog_task.write([0, 0, 0, 0])

    def close(self):
        """ Completely free the task.
            Close everything.
        """
        self.digital_task.stop()
        self.digital_task.close()
        
        self.analog_task.stop()
        self.analog_task.close()
#-------------------------------------------------------------------------------                                                    
    def timelapse_single(self, 
                        frequency_im = 0.0498,t_on_2p=0.005, 
                                                num_images=2 ):
        """ Acquire a time lapse and the shutter opens (single illumination) when the time lapse starts 
            - frequency_im = image acquisition frequency 
            
            - num_images = obvious.
        """
        assert(self.analog_channels ==[ 'ao3']), \
                "\n Select only channels  \'ao3\' for this modality"

        num_samples =  10**7
        num_pulses=1
        self.t_on_2p= t_on_2p
        # time variable calculated with the period of the im scan = 2* period
        #imaging !!! NB this is common for digital and analog
        self.t = np.linspace(0, 1/frequency_im, num_samples, dtype=np.float16)
        #------------------------Digital signal: 2P shutter---------------------
        # #time variable for 1 period of the scanning 
        self.t_tot=1/frequency_im
        # self.t_tot=1/frequency_im
        #duty calculated from time on and related to self.t so to the period of the scan
        self.duty_2p = self.t_on_2p/self.t_tot
        # self.duty_2p = self.t_on_2p/self.t_tot/2
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        self.trigger[:int(self.duty_2p*len(self.t))] += 1
        self.trigger *= (2**32 - 1)
        # print('f_im',frequency_im,'f_scan', frequency_im/2, 'duty', self.duty_2p, 't_tot', self.t_tot)
        #samples in one sec digital channel.  !!!NB same as analog channel
        samples_per_second_digital = int(len(self.t))/self.t_tot
       
        
        
        # Define the tasks to send the digiatl  signal, this signal starts when
        # the analog starts see source = clock 
        #!!! NB factor 2 due to imaging frequency 
        self.digital_task.timing.cfg_samp_clk_timing(rate= samples_per_second_digital ,
                       source='ao/SampleClock',
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))
        
        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        
        #----------------------------Analog signals: ls imaging-----------------
        # samples in one sec analog channel 
        samples_per_second_analog = int(len(self.t)*frequency_im) 
        
        print('samples/s ', samples_per_second_analog)
        self.signal_2 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_2 = 5 * (square(2 * np.pi * self.t  * frequency_im, duty = .1)+1)/2.
        
        signal_2[int(len(signal_2)*3/4+1):] = 0

        self.signal_2[:] = signal_2[:]
     
        self.matrix = np.zeros((1, len(self.signal_2)))
        # self.matrix[0,:] = self.signal_1
        self.matrix[0,:] = self.signal_2
        
        
        # the difference with before is in samps_per_chan
        # by design,eery triang wave goes up and down, so scans 2 images
        # that's why there is a 2 there
        # also the sample_mode is FINITE now
        self.analog_task.timing.cfg_samp_clk_timing(rate = samples_per_second_analog,\
                    sample_mode= nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=int(num_samples*num_images))

        analog_writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                    self.analog_task.out_stream, auto_start=False)

        digital_writer.write_many_sample_port_uint32(self.trigger)
        analog_writer.write_many_sample(self.matrix)
          
    
        #-----------------------------------------------------------------------
        #start -- NB order!Since the digital task waits for the analog clock we start
        #before the digital otherwise we have a delay betweeen the two tasks.
        
        # self.analog_task.start()
        # self.digital_task.start()
       
        self.digital_task.start()
        self.analog_task.start()
        
        # # Useful when you test this function 
        # self.analog_task.wait_until_done(float((1/frequency_im)*num_images+.05))
        # self.analog_task.stop()
        # self.digital_task.stop()
        print('\nScan done. \n')
        
        # plt.figure('plot')
        # plt.plot(self.t,self.trigger ,'y-')
        # plt.plot(self.t,self.signal_2,'b-')
        # plt.show()

#-------------------------------------------------------------------------------         
                    
    def timelapse_train(self, 
                        frequency_im = 0.0498,t_on_2p=0.005, \
                                                num_images=10, num_pulses=10 ):
        """ Acquire a time lapse and  the shutter opens (train illumination) when the time lapse starts. 
            - frequency_im = image acquisition frequency 
            - t_on_2p = time on for one pulse
            - num_images = num of images acquired.
            - num_pulses = num pulses for each image acquired
        """
        assert(self.analog_channels ==['ao3']), \
                "\n Select only channels  \'ao3\' for this modality"

        num_samples =  10**7
        self.t_on_2p = t_on_2p
        # # time variable calculated with the period of the im scan = 2* period
        # #imaging !!! NB this is common for digital and analog
        self.t = np.linspace(0, 1/frequency_im, num_samples, dtype=np.float16)
        # #------------------------Digital signal: 2P shutter-------------------
        #Period of one pulse 
        self.t_one_pulse=1/frequency_im
        # self.t_one_pulse=1/frequency_im/num_pulses
      
        #duty calculated from time on 
        self.duty_2p = self.t_on_2p/ self.t_one_pulse
        # self.duty_2p = self.t_on_2p/ self.t_one_pulse/2
        # print('f_im',frequency_im,'f_scan', frequency_im/2, 'duty', self.duty_2p, 't_tot', self.t_one_pulse)
        #freq one pulse
        self.frequency_one_pulse=1/ self.t_one_pulse
        #-----------------------------------------------------------------------
        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        #single square wave with scipy
        self.trigger = (2**32 - 1) * \
                 (square(2 * np.pi * self.t*self.frequency_one_pulse , self.duty_2p)+1)/2.
        
        self.trigger[int(len(self.trigger)*3/4+1):] = 0
    
        self.trigger = self.trigger.astype(np.uint32)
        #samples per second !!!NB same as analog channel
        samples_per_second_digital = int(len(self.t)*frequency_im)
        # samples_per_second_digital = int(len(self.t)*frequency_im/2)
             
        # Define the tasks to send the digiatl  signal, this signal starts when
        # the analog starts see source = clock , samples per channel as factor 
        # !!! NB 2 to have n_pulses for each image. 
        self.digital_task.timing.cfg_samp_clk_timing(rate= samples_per_second_digital,
                       source='ao/SampleClock',
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))
                       # samps_per_chan=int(num_samples*num_pulses))
        #                samps_per_chan=int(num_samples*num_pulses/2))
        # 
        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        
        #----------------------------Analog signals: ls imaging-----------------
        # samples in one sec analog channel 
        samples_per_second_analog = int(len(self.t)*frequency_im) 
        
 
        #step function imaging channel ao3 
        self.signal_2 = np.zeros((int(num_samples)), dtype=np.float16)
        signal_2 = 5 * (square(2 * np.pi * self.t  * frequency_im, duty = .1)+1)/2.
        signal_2[int(len(signal_2)*3/4+1):] = 0

        self.signal_2[:] = signal_2[:]
          
     
        self.matrix = np.zeros((1, len(self.signal_2)))
        # self.matrix[0,:] = self.signal_1
        self.matrix[0,:] = self.signal_2
    
        
        # Define analog tasks.
        # Every triang wave goes up and down, so scans 2 images,that's why there
        # is divided 2  
        
        self.analog_task.timing.cfg_samp_clk_timing(rate = samples_per_second_analog ,\
                    sample_mode = nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan = int(num_samples*num_images))

        analog_writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
                                    self.analog_task.out_stream, auto_start=False)
                                    
        # write digital and analog signals 
        digital_writer.write_many_sample_port_uint32(self.trigger)
        analog_writer.write_many_sample(self.matrix)
           
    
        #-----------------------------------------------------------------------
        #start -- NB order!Since the digital task waits for the analog clock we start
        #before the digital otherwise we have a delay betweeen the two tasks. 
        
    
        self.digital_task.start()
        self.analog_task.start()
        
        # self.analog_task.wait_until_done(float((1/frequency_im)*num_images+.05))
        # self.analog_task.stop()
        # self.digital_task.stop()
        print('\nScan done. \n')
        print('samples/s ', samples_per_second_analog, 'duty',self.duty_2p)
        

 
                        
    def train(self, 
                        frequency_im =0.05,t_on_2p=0.005, \
                                                num_images=10, num_pulses=10 ):
        """ Acquire a time lapse and  the shutter opens (train illumination) when the time lapse starts. 
            - frequency_im = image acquisition frequency 
            - t_on_2p = time on for one pulse
            - num_images = num of images acquired.
            - num_pulses = num pulses for each image acquired
        """
        assert(self.analog_channels ==['ao3']), \
                "\n Select only channels  \'ao3\' for this modality"

        num_samples =  10**5
        self.t_on_2p = t_on_2p
        # # time variable calculated with the period of the im scan = 2* period
        # #imaging !!! NB this is common for digital and analog
        self.t = np.linspace(0, 1/frequency_im, num_samples, dtype=np.float16)
        # #------------------------Digital signal: 2P shutter-------------------
        #Period of one pulse 
        self.t_one_pulse=1/frequency_im
        # self.t_one_pulse=1/frequency_im/num_pulses
        
        #duty calculated from time on 
        self.duty_2p = self.t_on_2p/ self.t_one_pulse
        # self.duty_2p = self.t_on_2p/ self.t_one_pulse/2
        # print('f_im',frequency_im,'f_scan', frequency_im/2, 'duty', self.duty_2p, 't_tot', self.t_one_pulse)
        #freq one pulse
        self.frequency_one_pulse=1/ self.t_one_pulse
        #-----------------------------------------------------------------------
        # Digital trigger for the shutter. Square wave.Value 2**32-1 because
        # the daq wants a uint32.
        self.trigger = np.zeros((int(num_samples)), dtype=np.uint32)
        #single square wave with scipy
        self.trigger = (2**32 - 1) * \
                 (square(2 * np.pi * self.t*self.frequency_one_pulse , self.duty_2p)+1)/2.
        
        self.trigger[int(len(self.trigger)*3/4+1):] = 0
    
        self.trigger = self.trigger.astype(np.uint32)
        #samples per second !!!NB same as analog channel
        samples_per_second_digital = int(len(self.t)*frequency_im)
        # samples_per_second_digital = int(len(self.t)*frequency_im/2)
             
        # Define the tasks to send the digiatl  signal, this signal starts when
        # the analog starts see source = clock , samples per channel as factor 
        # !!! NB 2 to have n_pulses for each image. 
        self.digital_task.timing.cfg_samp_clk_timing(rate= samples_per_second_digital,
                    #    source='ao/SampleClock',
                       sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                       samps_per_chan=int(num_samples*num_pulses))
                       # samps_per_chan=int(num_samples*num_pulses))
        #                samps_per_chan=int(num_samples*num_pulses/2))
        # 
        digital_writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
                                                   self.digital_task.out_stream, 
                                                   auto_start=False)
        
        #----------------------------Analog signals: ls imaging-----------------
        # samples in one sec analog channel 
        # samples_per_second_analog = int(len(self.t)*frequency_im) 
        
 
        # #step function imaging channel ao3 
        # self.signal_2 = np.zeros((int(num_samples)), dtype=np.float16)
        # signal_2 = 5 * (square(2 * np.pi * self.t  * frequency_im, duty = .1)+1)/2.
        # signal_2[int(len(signal_2)*3/4+1):] = 0

        # self.signal_2[:] = signal_2[:]
          
     
        # self.matrix = np.zeros((1, len(self.signal_2)))
        # # self.matrix[0,:] = self.signal_1
        # self.matrix[0,:] = self.signal_2
    
        
        # Define analog tasks.
        # Every triang wave goes up and down, so scans 2 images,that's why there
        # is divided 2  
        
        # self.analog_task.timing.cfg_samp_clk_timing(rate = samples_per_second_analog ,\
        #             sample_mode = nidaqmx.constants.AcquisitionType.FINITE,
        #             samps_per_chan = int(num_samples*num_images))

        # analog_writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(\
        #                             self.analog_task.out_stream, auto_start=False)
                                    
        # write digital and analog signals 
        digital_writer.write_many_sample_port_uint32(self.trigger)
        
           
    
        #-----------------------------------------------------------------------
        #start -- NB order!Since the digital task waits for the analog clock we start
        #before the digital otherwise we have a delay betweeen the two tasks. 
        
    
        self.digital_task.start()
     
        
        self.digital_task.wait_until_done(float((1/frequency_im)*num_images+.05))
      
        print('\nScan done. \n')

        plt.figure()
        plt.plot(self.trigger)
        # plt.ylim(-6, 6)
        plt.show()
                                
        