function testMatlabGui
% Requires MATLAB with App Designer and a graphical display.
% Invoke real callbacks and verify the rendered results.
addpath(fileparts(fileparts(mfilename('fullpath'))));
setupTorch;
a = interface1D;
cleanup = onCleanup(@()delete(a));
drawnow; pause(1);
f = a.Test1Button.ButtonPushedFcn; f(a.Test1Button,[]);
a.NumberofCellsEditField.Value = 20;
a.StoptimeEditField.Value = 0.01;
a.C_CflSlider.Value = 0.4;
f = a.RUNButton.ButtonPushedFcn; f(a.RUNButton,[]);
drawnow;
assert(startsWith(a.MessageAreaEditField.Value,'Completed'),a.MessageAreaEditField.Value);
line = findobj(a.UIAxes,'Type','line');
assert(numel(line)==1 && abs(line.XData(1)-0.025)<1e-12);
disp('PASS MATLAB launch, Sod, final frame and cell-center coordinates');
a.DisplayOptionDropDown.Value = 'Pressure';
f = a.LaststepButton.ButtonPushedFcn; f(a.LaststepButton,[]);
line = findobj(a.UIAxes,'Type','line'); before = line.YData;
a.GammaEditField.Value = 1.6;
f(a.LaststepButton,[]);line=findobj(a.UIAxes,'Type','line');
assert(isequal(before,line.YData));
a.WithChemicalReactionsCheckBox.Value = true;
f = a.WithChemicalReactionsCheckBox.ValueChangedFcn;f(a.WithChemicalReactionsCheckBox,[]);
a.p_l0Slider.Value=1e6;a.p_r0Slider.Value=101325;
f=a.p_l0Slider.ValueChangedFcn;f(a.p_l0Slider,[]);
f=a.p_r0Slider.ValueChangedFcn;f(a.p_r0Slider,[]);
a.StoptimeEditField.Value=1e-5;
f=a.RUNButton.ButtonPushedFcn;f(a.RUNButton,[]);
assert(startsWith(a.MessageAreaEditField.Value,'Completed'),a.MessageAreaEditField.Value);
a.DisplayOptionDropDown.Value='Lambda';
f=a.LaststepButton.ButtonPushedFcn;f(a.LaststepButton,[]);
line=findobj(a.UIAxes,'Type','line');assert(all(line.YData>=0 & line.YData<=1) && min(line.YData)<0.99);
f=a.AnimateStartButton.ButtonPushedFcn;f(a.AnimateStartButton,[]);
assert(abs(str2double(a.TimeTraker.Value)-1e-5)<1e-10);
disp('PASS MATLAB reactive SI case, lambda plot and animation');
a.rho_l0Slider.Value=0;
f=a.RUNButton.ButtonPushedFcn;f(a.RUNButton,[]);
assert(startsWith(a.MessageAreaEditField.Value,'Error:'));
assert(strcmp(a.RUNButton.Enable,'on'));
disp('PASS MATLAB invalid input handled and run button restored');
end
