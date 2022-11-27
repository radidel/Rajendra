filt = dsp.FIRFilter;
filt.Numerator = fircband(12,[0 0.4 0.5 1],[1 1 0 0],[1 0.2],... 
{'w' 'c'});
x = 0.1*randn(250,1);
n = 0.01*randn(250,1);
d = filt(x) + n;
mu = 0.8;
lms = dsp.LMSFilter(13,'StepSize',mu);

[y,e,w] = lms(x,d);
plot(1:250, [d,y,e]);
title('System Identification of an FIR filter');
legend('Desired','Output','Error');
xlabel('Time index');
ylabel('Signal value');

stem([(filt.Numerator).' w])
title('System Identification by Adaptive LMS Algorithm')
legend('Actual filter weights','Estimated filter weights',...
       'Location','NorthEast')