function hfig = fPatchOverlay( dImg, dPatch, iScale, dAlpha, sPathOut, cPlotLimits, lLabel, lGray, lRot)
%FPATCHOVERLAY overlay figure   
    
    if (nargin < 9)
        lRot=false;
    end
    if(nargin < 8)
        lGray = false;
    end
    if(nargin < 7)
        lLabel = true;
    end
    if(nargin < 6)
        xLimits = [1 size(dImg,2)];
        yLimits = [1 size(dImg,1)];
    else
        xLimits = cPlotLimits{1};
        yLimits = cPlotLimits{2};
    end
    if(nargin < 5)
        sPathOut = cd;
    end
    if(nargin < 4)
        dAlpha = 0.6;
    end
    if(nargin < 3)
        iScale = [0 1; 0 1];
    end
    if ~isa(dPatch, 'cell') dPatch={dPatch}; end
        
        
   

    h.sPathOut = sPathOut;
    h.dAlpha = dAlpha;
    h.lGray = lGray;
    h.colRange = iScale;
    
    h.model=1;
    
    h.WindowCenter=0.5;
    h.WindowWidth=1;
    h.oldMousePos=[0 0];
    h.MouseExtendActive=false;
    
	hfig = figure;
    hold on
    %hold on fixed the axes problem, but no idea why...
    

 
    dImg = ((dImg - min(dImg(:))).*(h.colRange(1,2)-h.colRange(1,1)))./(max(dImg(:) - min(dImg(:)) ));%falsch???
    %divides by two...
    
    if(h.lGray)%???????????????TODO         
        alpha = bsxfun(@times, ones(size(dPatch,1), size(dPatch,2)), .6);

        % find a scale dynamically with some limit
        Foreground_min = min( min(dPatch(:)), h.colRange(1) );
        Foreground_max = max( max(dPatch(:)), h.colRange(2) );
        Background_blending = bsxfun(@times, dImg, bsxfun(@minus,1,alpha));
        Foreground_blending = bsxfun( @times, bsxfun( @rdivide, ...
            bsxfun(@minus, dPatch, Foreground_min), ... 
            Foreground_max-Foreground_min ), alpha );
        h.dImg = Background_blending + Foreground_blending;
        h.hI = imshow(h.dImg(:,:,1), h.colRange); 
    else
        h.hI = axes();
        h.dImg = dImg;
        h.dPatch = dPatch;
        [h.hFront,h.hBack] = imoverlay(dImg(:,:,1,1),dPatch{h.model}(:,:,1,1),h.colRange(1,:),h.colRange(2,:),'jet',h.dAlpha, h.hI);
    end
    
    xlim(xLimits);
    ylim(yLimits);
    h.hA = gca;
    h.iActive = 1;
    if(lLabel)%modelLabel TODO
%         h.hT = uicontrol('Style','text', 'units','normalized', 'Position', [0.925 0.975 0.075 0.0255],'String',sprintf('I: [%.2f:%.2f]', h.colRange(1), h.colRange(2)),'ForegroundColor','k','Backgroundcolor','w');
        h.hT = uicontrol('Style','text', 'units','normalized', 'Position', [0.925 0.975 0.075 0.0255],'String',...
            sprintf('%02d/%02d', h.iActive, size(h.dImg,3)),'ForegroundColor','k','Backgroundcolor','w');
        h.hModel = uicontrol('Style','text', 'units','normalized', 'Position', [0.925 0.95 0.075 0.0255],'String',...
            sprintf('M:%02d/%02d',h.model, length(h.dPatch)),'ForegroundColor','k');
    end
    h.lLabel = lLabel;
    h.lRot=lRot;%rotate the pictures like the patches predicted...
    h.layer=0;
    set(h.hA, 'Position', [0 0 1 1]);
    %pos=[0, 0, size(dImg, 2).*4, size(dImg, 1).*4];
    set(hfig, 'Position', [0 0 size(dImg, 2).*4 size(dImg, 1).*4]);
    set(hfig, 'WindowScrollWheelFcn', @fScroll);
    set(hfig, 'KeyPressFcn'         , @fKeyPressFcn);
    set(hfig, 'WindowButtonMotionFcn'         , @fWindowFcn);
    set(hfig, 'WindowButtonDownFcn'         , @fWindowButtonDownFcn);
    set(hfig, 'WindowButtonUpFcn', @fWindowButtonUpFcn);
    colormap(jet(100));
    currpath = fileparts(mfilename('fullpath'));
    addpath(genpath([fileparts(fileparts(currpath)),filesep,'export_fig']));
    
    movegui('center');
    hold on
    
    guidata(hfig, h);
end

function fScroll(hObject, eventdata, handles)

    h = guidata(hObject);
    if eventdata.VerticalScrollCount < 0
        h.iActive = max([1 h.iActive - 1]);
    else
        h.iActive = min([size(h.dImg, 3) h.iActive + 1]);
    end

    if(h.lGray)
        set(h.hI, 'CData', h.dImg(:,:,h.iActive));
    else
        %[hFront,hBack] = imoverlay(dImg(:,:,31,1),dProbOverlay(:,:,31,1),[0.32903,1.5],[0,0.57486],'parula2',0.60004);
         [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                        [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
    end
    set(h.hT, 'String', sprintf('%02d/%02d', h.iActive, size(h.dImg,3)));%eg 08/40
    
    guidata(hObject, h);
end

function fKeyPressFcn(hObject, eventdata)
    h=guidata(hObject);
    digits=linspace(0,9,10);
    if(strcmpi(eventdata.Key,'p'))
        h = guidata(hObject);
        set(h.hT, 'Visible', 'off');
        set(h.hModel, 'Visible', 'off');
        set(h.hI, 'Visible', 'off');
        if(~exist(h.sPathOut,'dir'))
            mkdir(h.sPathOut);
        end
        sFiles = dir(h.sPathOut);
        iFound = cellfun(@(x) ~isempty(x), regexp({sFiles(:).name},[num2str(h.iActive,'%03d')]));
        if(any(iFound))
            sFile = [num2str(h.iActive,'%03d'),'_',num2str(nnz(iFound))];
        else
            sFile = num2str(h.iActive,'%03d');
        end
%         iFile = nnz(~cell2mat({sFiles(:).isdir})) + 1;
%         sFile = num2str(iFile);
        try
            export_fig([h.sPathOut,filesep,sFile,'.tif']);
        catch
            warning('export_fig() not on path');
        end
        set(h.hT, 'Visible', 'on');
        set(h.hModel, 'Visible', 'on');
    elseif(any(arrayfun(@(x) strcmpi(eventdata.Key,num2str(x)), digits)))
        
        h.layer=h.layer*10+ eventdata.Key-48;% 0 is asciicode 48...
    elseif(strcmpi(eventdata.Key, 'return'))%goto specified layer
        if(h.layer<=size(h.dImg,3) && ~h.layer==0)
            if(h.lGray)
                set(h.hI, 'CData', h.dImg(:,:,h.layer));
            else
                 if(h.iActive==h.layer) return; end
                 h.iActive=h.layer;
                 [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                        [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
            end
            set(h.hT, 'String', sprintf('%02d/%02d', h.layer, size(h.dImg,3)));%eg 08/40
            
            
        end
        h.layer=0;
    elseif(strcmpi(eventdata.Key, 'rightarrow'))
        if (h.dAlpha<=0.8)
            h.dAlpha=h.dAlpha +0.1999;
             [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                        [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
        end
    elseif(strcmpi(eventdata.Key, 'leftarrow'))
        if (h.dAlpha>=0.2)
            h.dAlpha=h.dAlpha -0.1999;
             [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                         [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
        end
    elseif(strcmpi(eventdata.Key, 'downarrow'))
        if h.model>1
            h.model=h.model -1;
            if (h.lRot)
                 [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                         [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
              
            else
                 [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                        [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
            end
            set(h.hModel, 'String', sprintf('M:%02d/%02d', h.model, length(h.dPatch)));
        end
    elseif(strcmpi(eventdata.Key, 'uparrow'))
        if h.model<length(h.dPatch)
            h.model=h.model +1;
            if (h.lRot)
                [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                        [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
            else
                [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),...
                        h.dPatch{h.model}(:,:,h.iActive,1),...
                        [h.colRange(1,:)],...
                        [h.WindowCenter- h.WindowWidth /2, h.WindowCenter + h.WindowWidth /2],...
                        'jet',h.dAlpha, h.hI);
            end
            set(h.hModel, 'String', sprintf('M:%02d/%02d', h.model, length(h.dPatch)));
        end
    end
    guidata(hObject, h);
end

function fWindowFcn(hObject, eventdata)
h=guidata(hObject);
if h.MouseExtendActive
    if ~(h.oldMousePos==[0,0])
        tuner=0.005;
        MouseDiff= hObject.CurrentPoint - h.oldMousePos;
        set(h.hModel, 'String', sprintf('%0.0f,%0.0f', MouseDiff(1), MouseDiff(2)));
        h.WindowCenter = h.WindowCenter + tuner*MouseDiff(1);
        h.WindowWidth = h.WindowWidth + tuner*MouseDiff(2);
        
        set(h.hBack,'CData' , repmat(mat2gray(double(h.dImg(:,:,h.iActive,1)),...
            double([h.WindowCenter-h.WindowWidth/2, h.WindowCenter+h.WindowWidth/2])),[1,1,3]));
        % Display the back image
    end
    h.oldMousePos = hObject.CurrentPoint;
else
    set(h.hModel, 'String', sprintf('%0.0f/%0.0f',hObject.CurrentPoint(1),hObject.CurrentPoint(2)));
end
guidata(hObject, h);
end

function fWindowButtonDownFcn(hObject, eventdata)
    if strcmpi(eventdata.Source.SelectionType, 'extend')
        h=guidata(hObject);
        h.MouseExtendActive=true;
        guidata(hObject, h);
    end
end
function fWindowButtonUpFcn(hObject, eventdata)
    h=guidata(hObject);
    h.MouseExtendActive=false;
    h.oldMousePos=[0,0];
    guidata(hObject, h);
end

