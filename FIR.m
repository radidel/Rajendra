%% Start
clc;
clear all;
close all;
impL    = 400;
Fs      = 8000;
Fcutoff = 2000;
imp     = fir1(impL,2*Fcutoff/Fs);
fdfOA = dsp.FrequencyDomainFIRFilter(imp,'Method','overlap-add');
fdfOS = dsp.FrequencyDomainFIRFilter(imp,'Method','overlap-save');
fir = dsp.FIRFilter('Numerator',imp);
dly = dsp.Delay('Length',fdfOA.Latency);
frameLen  = 400;

sin_100Hz = dsp.SineWave('Frequency',100,'SampleRate',Fs,...
    'SamplesPerFrame',frameLen);
sin_3KHz  = dsp.SineWave('Frequency',3e3,'SampleRate',Fs,...
    'SamplesPerFrame',frameLen);
ts = timescope('TimeSpanOverrunAction','Scroll',...
    'ShowGrid',true,'TimeSpanSource','Property','TimeSpan',5 * frameLen/Fs,...
    'YLimits',[-1.1 1.1],...
    'ShowLegend',true,...
    'SampleRate',Fs,...
    'ChannelNames',{'Overlap-add','Overlap-save','Direct-form FIR'});

numFrames = 1e4;
for idx = 1:numFrames
    x = sin_100Hz() + sin_3KHz() + 0.01*randn(frameLen,1);
    yOA = fdfOA(x);
    yOS = fdfOS(x);
    yFIR = fir(dly(x));
    ts([yOA,yOS,yFIR]);
end

impL    = 400;
Fs      = 8000;
Fcutoff = 2000;
imp     = fir1(impL,2 * Fcutoff / Fs);
H   = fft(imp , 2 * numel(imp));
oa  = dsp.FrequencyDomainFIRFilter('NumeratorDomain','Frequency',...
    'FrequencyResponse', H,...
    'NumeratorLength',numel(imp),...
    'Method','overlap-add');
fprintf('Frequency domain filter latency is %d samples\n',oa.Latency);

















