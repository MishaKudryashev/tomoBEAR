classdef GCTFCtfphaseflipCTFCorrection < Module
    methods
        function obj = GCTFCtfphaseflipCTFCorrection(configuration)
            obj = obj@Module(configuration);
        end
        
        function obj = setUp(obj)
            obj = setUp@Module(obj);
            createStandardFolder(obj.configuration, "ctf_corrected_aligned_tilt_stacks_folder", false);
            createStandardFolder(obj.configuration, "ctf_corrected_binned_aligned_tilt_stacks_folder", false);
        end
        
        function obj = process(obj)
            disp("INFO: **STARTING PARAMETERS**");
            disp("INFO: Input Folder: " + obj.input_path);
            disp("INFO: Output Folder: " + obj.output_path);
            % NOTE:TODO: it's probably the safest to trust the apix value in the
            % configuration for unbinned data
            field_names = fieldnames(obj.configuration.tomograms);
            if isfield(obj.configuration, "apix")
                apix = obj.configuration.apix * obj.configuration.ft_bin;
            elseif obj.configuration.tomograms.(field_names{obj.configuration.set_up.j}).apix == 0
                folder_contents = getOriginalMRCs(obj.configuration);
                disp("INFO: determining pixel size from header");
                command = sprintf("header %s | grep Pixel", string(folder_contents(1).folder + string(filesep) + folder_contents(1).name));
                output = executeCommand(command, false, obj.log_file_id);
                printVariable(output);
                
                matching_results = regexp(output, "(\d+.\d+)", "match");
                obj.configuration.tomograms.(field_names{obj.configuration.set_up.j}).apix = str2double(matching_results{1});
                apix = str2double(matching_results{1} * obj.configuration.ft_bin);
            else
                disp("INFO: taking pixel size from configuration");
                apix = obj.configuration.tomograms.(field_names{obj.configuration.set_up.j}).apix * obj.configuration.ft_bin;
                printVariable(apix);
            end
            tilt_files = getFilesFromLastModuleRun(obj.configuration,"AreTomo","tlt","last");
            if ~isempty(tilt_files)
                tilt_files{1} = strrep(tilt_files{1}, "._", "_");
            elseif isempty(tilt_files)
                if obj.configuration.use_rawtlt == true
                    tilt_files = getFilePathsFromLastBatchruntomoRun(obj.configuration, "rawtlt");
                else
                    tilt_files = getFilePathsFromLastBatchruntomoRun(obj.configuration, "tlt");
                end
            end
            if obj.configuration.use_aligned_stack == true
                if ~isempty(getFilePathsFromLastBatchruntomoRun(obj.configuration, "ali"))
                    tilt_stacks = getFilePathsFromLastBatchruntomoRun(obj.configuration, "ali");
                elseif ~isempty(getFilesFromLastModuleRun(obj.configuration,"AreTomo","ali","last"))
                    tilt_stacks = getFilesFromLastModuleRun(obj.configuration,"AreTomo","ali","last");
                else
                    error("ERROR: Aligned stack was requested to use, but was not found!");
                end
            else
                tilt_stacks = getTiltStacksFromStandardFolder(obj.configuration, true);
                tilt_stacks = tilt_stacks(contains({tilt_stacks(:).name}, sprintf("%s_%03d", obj.configuration.tomogram_output_prefix, obj.configuration.set_up.j)));
            end
            [path, name, extension] = fileparts(tilt_files{1});
            tilt_index_angle_mapping = sort(obj.configuration.tomograms.(obj.name).tilt_index_angle_mapping(2,:));
            for i = 1:length(tilt_stacks)
                [path, name, extension] = fileparts(tilt_files{i});
                if iscell(tilt_stacks)
                    [folder, tilt_stack_name, tilt_stack_extension] = fileparts(tilt_stacks{i});
                else
                    [folder, tilt_stack_name, tilt_stack_extension] = fileparts(tilt_stacks(i).folder + string(filesep) + tilt_stacks(i).name);
                end
                destination_folder = obj.output_path;
                slice_folder = destination_folder + string(filesep) + obj.configuration.slice_folder;
                obj.dynamic_configuration.defocus_slice_folder_path = slice_folder;
                [status_mkdir, message, message_id] = mkdir(slice_folder);
                defocus_file_destination = destination_folder + string(filesep) + obj.name + ".defocus";
                defocus_file_id = fopen(defocus_file_destination, "w");
                tilt_file_destination = destination_folder + string(filesep) + obj.name + ".tlt";
                createSymbolicLink(tilt_files{i}, tilt_file_destination, obj.log_file_id);
                tilt_file_id = fopen(tilt_file_destination, "r");
                if tilt_file_id == -1
                    obj.status = 0;
                end
%                 % NOTE: use the raw stack if aligned stack binning is
%                 % higher than 1
%                 if (isfield(obj.configuration, "use_aligned_stack") && obj.configuration.use_aligned_stack == false) || obj.configuration.aligned_stack_binning > 1%) && isempty(getFilesFromLastModuleRun(obj.configuration,"AreTomo","tlt","last"))
%                     xf_file_destination = destination_folder + string(filesep) + obj.name + ".xf";
%                     output = createSymbolicLink(xf_files{i}, xf_file_destination, obj.log_file_id);
%                 end
                disp("INFO: splitting " + tilt_stack_name + "...");
                if iscell(tilt_stacks)
                    source = tilt_stacks{i};
                else
                    source = string(tilt_stacks(i).folder) + string(filesep) + string(tilt_stacks(i).name);
                end
                % TODO: introduce checks
                [status_mkdir, message, message_id] = mkdir(destination_folder);
                destination = destination_folder + string(filesep) + tilt_stack_name;
                output = createSymbolicLink(source, destination, obj.log_file_id);
                output = executeCommand("newstack -split 1 -append mrc "...
                    + destination + " "...
                    + slice_folder + string(filesep) + obj.name...
                    + "_" + obj.configuration.slice_suffix + "_", false, obj.log_file_id);
                return_folder = cd(slice_folder);
                disp("INFO: **GCTF ESTIMATION**");
                if isfield(obj.configuration, "nominal_defocus_in_nm") && obj.configuration.nominal_defocus_in_nm ~= 0
                    % TODO: could be also 2 numbers for lower and upper
                    % limit or factors in different variable names
                    lower_l = round(obj.configuration.nominal_defocus_in_nm / obj.configuration.defocus_limit_factor) * 10^4;
                    upper_l = round(obj.configuration.nominal_defocus_in_nm * obj.configuration.defocus_limit_factor) * 10^4;
                else
                    if isfield(obj.configuration, "apix")
                        apix = obj.configuration.apix * obj.configuration.ft_bin;
                    else
                        apix = obj.configuration.greatest_apix * obj.configuration.ft_bin;
                    end
                    disp("INFO: Starting Gctf estimation on " + obj.name + "!");
                    command = "CUDA_VISIBLE_DEVICES=" + (obj.configuration.set_up.gpu - 1) + " " +obj.configuration.ctf_correction_command...
                        + " --apix " + apix...
                        + " "...
                        + obj.name + "_" + obj.configuration.slice_suffix + "_";
                    if isfield(obj.configuration, "tilt_index_angle_mapping") && isfield(obj.configuration.tilt_index_angle_mapping, obj.name)
                        if max(obj.configuration.tilt_index_angle_mapping.(obj.name)(4,:)) < 10
                            file = sprintf("%d", obj.configuration.tilt_index_angle_mapping.(obj.name)(4,tilt_index_angle_mapping == 0));
                        else
                            file = sprintf("%02d", obj.configuration.tilt_index_angle_mapping.(obj.name)(4,tilt_index_angle_mapping == 0));
                        end
                    else
                        if max(obj.configuration.tomograms.(obj.name).tilt_index_angle_mapping(4,:)) < 10
                            file = sprintf("%d", obj.configuration.tomograms.(obj.name).tilt_index_angle_mapping(4,tilt_index_angle_mapping == 0));
                        else
                            file = sprintf("%02d", obj.configuration.tomograms.(obj.name).tilt_index_angle_mapping(4,tilt_index_angle_mapping == 0));
                        end
                    end
                    command = command + file + ".mrc";
                    output = executeCommand(command, false, obj.log_file_id);
                    delete(obj.output_path + string(filesep) + obj.configuration.slice_folder + string(filesep) + obj.name + "_" + obj.configuration.slice_suffix + "_"...
                        + file...
                        + "_EPA.log");
                    delete(obj.output_path + string(filesep) + obj.configuration.slice_folder + string(filesep) + "micrographs_all_gctf.star");
                    delete(obj.output_path + string(filesep) + obj.configuration.slice_folder + string(filesep) + obj.name + "_" + obj.configuration.slice_suffix + "_"...
                        + file...
                        + "_gctf.log")
                    delete(obj.output_path + string(filesep) + obj.configuration.slice_folder + string(filesep) + obj.name + "_" + obj.configuration.slice_suffix + "_"...
                        + file...
                        + ".ctf")
                    line_divided_text = textscan(output, '%s', 'delimiter', '\n');
                    final_values = line_divided_text{1}{contains(line_divided_text{1}, "Final Values")};
                    final_values_splitted = strsplit(final_values);
                    global_defocus_1_in_angstrom = str2double(final_values_splitted{1});
                    global_defocus_2_in_angstrom = str2double(final_values_splitted{2});
                    global_defocus_average_in_angstrom = (global_defocus_1_in_angstrom + global_defocus_2_in_angstrom) / 2;
                    % TODO: use cosine for defocus interval to be used for
                    % optimization
                    lower_l = global_defocus_average_in_angstrom / 2;
                    upper_l = global_defocus_average_in_angstrom * 1.5;
                    if ~isfield(obj.dynamic_configuration, "global_lower_defocus_average_in_angstrom")
                        obj.dynamic_configuration.global_lower_defocus_average_in_angstrom = lower_l;
                        obj.dynamic_configuration.global_upper_defocus_average_in_angstrom = upper_l;
                    else
                        obj.dynamic_configuration.global_lower_defocus_average_in_angstrom = obj.dynamic_configuration.global_lower_defocus_average_in_angstrom + lower_l / 2;
                        obj.dynamic_configuration.global_upper_defocus_average_in_angstrom = obj.dynamic_configuration.global_uppper_defocus_average_in_angstrom + upper_l / 2;
                    end
                end
                disp("INFO: CALCULATED LIMITS: Lower Limit: " + lower_l + " Upper Limit: " + upper_l);
                command = "CUDA_VISIBLE_DEVICES=" + (obj.configuration.set_up.gpu - 1) + " " + string(obj.configuration.ctf_correction_command)...
                    + " --apix " + apix...
                    + " --defL " + lower_l...
                    + " --defH " + upper_l...
                    + " --astm " + obj.configuration.estimated_astigmatism;
                if obj.configuration.do_phase_flip == true
                    disp("INFO: tilt stacks will be CTF-corrected by GCTF.");
                    command = command + " --do_phase_flip";
                end
                if obj.configuration.do_EPA == true
                    command = command + " --do_EPA";
                end
                command = command + " " + obj.name + "*.mrc";
                output = executeCommand(command, false, obj.log_file_id);
                
                if obj.configuration.do_phase_flip == true
                    % NOTE: assembling CTF corrected stack
                    if tilt_stack_extension == ""
                        tilt_stack_extension = ".st";
                    end
                    ctf_corrected_stack_destination = obj.output_path...
                        + string(filesep) + tilt_stack_name...
                        + "_" + obj.configuration.ctf_corrected_stack_suffix...
                        + tilt_stack_extension;
                    tilt_views_ali_ctfc = dir(obj.name + "*_pf.mrc");
                    tilt_views_ali_ctfc_list = string([]);
                    for j=1:length(tilt_views_ali_ctfc)
                        tilt_views_ali_ctfc_list(j) = tilt_views_ali_ctfc(j).folder + string(filesep) + tilt_views_ali_ctfc(j).name; 
                    end    
                    output = executeCommand("newstack "...
                        + strjoin(tilt_views_ali_ctfc_list, " ") + " "...
                        + ctf_corrected_stack_destination, false, obj.log_file_id);
                    
                    % NOTE: linking CTF corrected stack
                    if contains(tilt_stack_name, "_bin_")
                        tilt_stack_name_suffix = "";
                        splitted_tilt_stack_name = split(tilt_stack_name, "_bin_");
                        tilt_stack_ali_bin = str2double(splitted_tilt_stack_name(end));
                    else
                        if obj.configuration.use_aligned_stack == true
                            tilt_stack_ali_bin = obj.configuration.aligned_stack_binning;
                        else
                            tilt_stack_ali_bin = 1;
                        end
                        tilt_stack_name_suffix = "_bin_" + num2str(tilt_stack_ali_bin);
                    end
                    filename_link_destination = tilt_stack_name + tilt_stack_name_suffix + ".ali";

                    if tilt_stack_ali_bin == 1
                        folder_destination = obj.configuration.ctf_corrected_aligned_tilt_stacks_folder;
                    else
                        folder_destination = obj.configuration.ctf_corrected_binned_aligned_tilt_stacks_folder;
                    end

                    path_destination = obj.configuration.processing_path + filesep + obj.configuration.output_folder + filesep + folder_destination + filesep + obj.name;
                    link_destination = path_destination + filesep + filename_link_destination;
                    
                    if exist(path_destination, "dir")
                        rmdir(path_destination, "s");
                    end
                    mkdir(path_destination);
                    createSymbolicLink(ctf_corrected_stack_destination, link_destination, obj.log_file_id);
                end
                
                view_list = dir(slice_folder + string(filesep) + obj.name + "_*_" + "gctf.log");
                if obj.configuration.defocus_file_version <= 2
                    j_length = length(view_list);
                else
                    j_length = length(view_list) + 1;
                end
                for j = 1:j_length
                    if obj.configuration.defocus_file_version <= 2
                        gctf_obj.log_file_id = fopen(view_list(j).folder + string(filesep) + view_list(j).name, "r");
                        line_divided_text = textscan(gctf_obj.log_file_id, "%s", "delimiter", "\n");
                        final_values = line_divided_text{1}{contains(line_divided_text{1}, "Final Values")};
                        final_values_splitted = strsplit(final_values);
                        local_defocus_1_in_angstrom = str2double(final_values_splitted{1});
                        local_defocus_2_in_angstrom = str2double(final_values_splitted{2});
                        astigmatism_angle = str2double(final_values_splitted{3});
                        local_defocus_1_in_nanometers = local_defocus_1_in_angstrom / 10;
                        local_defocus_2_in_nanometers = local_defocus_2_in_angstrom / 10;
                        if local_defocus_1_in_nanometers > local_defocus_2_in_nanometers
                            local_defocus_in_nanometers_temporary = local_defocus_2_in_nanometers;
                            local_defocus_2_in_nanometers = local_defocus_1_in_nanometers;
                            local_defocus_1_in_nanometers = local_defocus_in_nanometers_temporary;
                            astigmatism_angle = astigmatism_angle - 90;
                        end
                        tilt_angle = fgetl(tilt_file_id);
                        if j == 1
                            fwrite(defocus_file_id, sprintf("%s %s %s %s %s %s %s %s\n", num2str(j), num2str(j), num2str(tilt_angle), num2str(tilt_angle), num2str(local_defocus_1_in_nanometers), num2str(local_defocus_2_in_nanometers), num2str(astigmatism_angle), num2str(configuration.defocus_file_version)));
                        else
                            fwrite(defocus_file_id, sprintf("%s %s %s %s %s %s %s\n", num2str(j), num2str(j), num2str(tilt_angle), num2str(tilt_angle), num2str(local_defocus_1_in_nanometers), num2str(local_defocus_2_in_nanometers), num2str(astigmatism_angle)));
                        end
                    else
                        if j == 1
                            fwrite(defocus_file_id, sprintf("%s 0 0. 0. 0 %s\n", num2str(obj.configuration.defocus_file_version_3_flag), num2str(obj.configuration.defocus_file_version)));
                        else
                            gctf_obj.log_file_id = fopen(view_list(j-1).folder + string(filesep) + view_list(j-1).name, "r");
                            line_divided_text = textscan(gctf_obj.log_file_id, "%s", "delimiter", "\n");
                            final_values = line_divided_text{1}{contains(line_divided_text{1}, "Final Values")};
                            final_values_splitted = strsplit(final_values);
                            local_defocus_1_in_angstrom = str2double(final_values_splitted{1});
                            local_defocus_2_in_angstrom = str2double(final_values_splitted{2});
                            astigmatism_angle = str2double(final_values_splitted{3});
                            local_defocus_1_in_nanometers = local_defocus_1_in_angstrom / 10;
                            local_defocus_2_in_nanometers = local_defocus_2_in_angstrom / 10;
                            
                            if local_defocus_1_in_nanometers > local_defocus_2_in_nanometers
                                local_defocus_in_nanometers_temporary = local_defocus_2_in_nanometers;
                                local_defocus_2_in_nanometers = local_defocus_1_in_nanometers;
                                local_defocus_1_in_nanometers = local_defocus_in_nanometers_temporary;
                                astigmatism_angle = astigmatism_angle - 90;
                            end
                            tilt_angle = fgetl(tilt_file_id);
                            fwrite(defocus_file_id, sprintf("%s %s %s %s %s %s %s\n", num2str(j-1), num2str(j-1), num2str(tilt_angle), num2str(tilt_angle), num2str(local_defocus_1_in_nanometers), num2str(local_defocus_2_in_nanometers), num2str(astigmatism_angle)));
                        end
                    end
                end
                fclose(defocus_file_id);
                fclose(tilt_file_id);
                cd(return_folder);
            end
            
            if obj.configuration.run_ctf_phase_flip == true
                disp("INFO: tilt stacks will be CTF-corrected by Ctfphaseflip.");
                % NOTE: run Ctfphaseflip CTF-correction always on aligned stack
                if ~isempty(getFilePathsFromLastBatchruntomoRun(obj.configuration, "ali"))
                    tilt_stacks_ali = getFilePathsFromLastBatchruntomoRun(obj.configuration, "ali");
                elseif ~isempty(getFilesFromLastModuleRun(obj.configuration,"AreTomo","ali","last"))
                    tilt_stacks_ali = getFilesFromLastModuleRun(obj.configuration,"AreTomo","ali","last");
                else
                    error("ERROR: Aligned stacks are required to perform CTF-correction using Ctfphaseflip, but were not found!");
                end
                for i = 1:length(tilt_stacks)
                    if iscell(tilt_stacks_ali)
                        tilt_stack_ali_full_path = tilt_stacks_ali{i};
                    else
                        tilt_stack_ali_full_path = tilt_stacks_ali(i).folder + string(filesep) + tilt_stacks_ali(i).name;
                    end
                    [~, tilt_stack_ali_name, tilt_stack_ali_ext] = fileparts(tilt_stack_ali_full_path);
                    if tilt_stack_ali_ext == ""
                        tilt_stack_ali_ext = ".st";
                    end
                    ctf_corrected_stack_destination = obj.output_path...
                        + string(filesep) + tilt_stack_ali_name...
                        + "_" + obj.configuration.ctf_corrected_stack_suffix...
                        + tilt_stack_ali_ext;
                    command = "ctfphaseflip -input " + tilt_stack_ali_full_path...
                        + " -output " + ctf_corrected_stack_destination...
                        + " -angleFn " + tilt_file_destination...
                        + " -defFn " + defocus_file_destination...
                        + " -defTol " + obj.configuration.defocus_tolerance...
                        + " -iWidth " + obj.configuration.iWidth...
                        + " -maxWidth " + obj.configuration.maximum_strip_width...
                        + " -pixelSize " + apix...
                        + " -volt " + obj.configuration.keV...
                        + " -cs " + obj.configuration.spherical_aberation...
                        + " -ampContrast " + obj.configuration.ampContrast;
                    if obj.configuration.use_aligned_stack == false
                        % NOTE: if CTF was estimated on raw stack, write
                        % and use XF transform file while correcting CTF
                        % TODO: find better way to make checks
                        % (e.g. link xf files to separate folder)
                        xf_files = getFilesFromLastModuleRun(obj.configuration,"AreTomo","xf","last");
                        if ~isempty(xf_files)
                            xf_files = xf_files{1};
                        else
                            xf_files = getFilesFromLastModuleRun(obj.configuration,"AreTomo","aln","last");
                            if isempty(xf_files)
                                xf_files = getFilePathsFromLastBatchruntomoRun(obj.configuration, "xf");
                                xf_files = xf_files{1};
                            else
                                fid_in = fopen(xf_files{1});
                                lines_in_cells = textscan(fid_in, "%s","Delimiter","\n");
                                fclose(fid_in);
                                fid_out = fopen(obj.output_path + filesep + obj.name + ".xf", "w+");
                                for j = 4:length(lines_in_cells{1})
                                    numbers_in_line = textscan(lines_in_cells{1}{j}, "%f %f %f %f %f %f %f %f %f %f");
                                    rotation_matrix = rotz(numbers_in_line{2});
                                    fprintf(fid_out, "%f %f %f %f %f %f\n", rotation_matrix(1,1), rotation_matrix(1,2), rotation_matrix(2,1), rotation_matrix(2,2), numbers_in_line{4}, numbers_in_line{5});
                                end
                                fclose(fid_out);
                                xf_files = obj.output_path + filesep + obj.name + ".xf";
                            end
                        end
                        xf_file_destination = destination_folder + string(filesep) + obj.name + ".xf";
                        createSymbolicLink(xf_files, xf_file_destination, obj.log_file_id);

                        command = command + " -xform " + xf_file_destination;
                    end
                    % TODO: get number as numeric for better version check
                    if obj.configuration.set_up.gpu > 0 && versionGreaterThan(obj.configuration.environment_properties.imod_version, "4.10.9")
                        command = command + " -gpu " + obj.configuration.set_up.gpu;
                    end
                    executeCommand(command, false, obj.log_file_id);
                    
                    if contains(tilt_stack_ali_name, "bin_")
                        filename_link_destination = tilt_stack_ali_name + ".ali";
                        splitted_tilt_stack_ali_name = split(tilt_stack_ali_name, "_bin_");
                        tilt_stack_ali_bin = str2double(splitted_tilt_stack_ali_name(end));
                    else
                        filename_link_destination = tilt_stack_ali_name + "_bin_" + num2str(obj.configuration.aligned_stack_binning) + ".ali";
                        tilt_stack_ali_bin = obj.configuration.aligned_stack_binning;
                    end
                    
                    if tilt_stack_ali_bin == 1
                        folder_destination = obj.configuration.ctf_corrected_aligned_tilt_stacks_folder;
                    else
                        folder_destination = obj.configuration.ctf_corrected_binned_aligned_tilt_stacks_folder;
                    end
                    
                    path_destination = obj.configuration.processing_path + filesep + obj.configuration.output_folder + filesep + folder_destination + filesep + obj.name;
                    link_destination = path_destination + filesep + filename_link_destination;
                    
                    if exist(path_destination, "dir")
                        rmdir(path_destination, "s");
                    end
                    mkdir(path_destination);
                    createSymbolicLink(ctf_corrected_stack_destination, link_destination, obj.log_file_id);
                end
            end
            % NOTE: DISABLED RECONSTRUCTION HERE (USE RECONSTRUCT MODULE!)
            % NOTE: run Tilt reconstruction always on CTF-corrected stack
%             if obj.configuration.reconstruct_tomograms == true
%                 disp("INFO: tomograms will be generated.");
%                 ctf_corrected_tilt_stacks = ...
%                 for i = 1:length(ctf_corrected_tilt_stacks)
%                     ctf_corrected_stack_destination
%                         ctf_corrected_tomogram_destination = tilt_stack_ali_path + "_"...
%                             + obj.configuration.ctf_corrected_stack_suffix + "_"...
%                             + obj.configuration.tomogram_suffix + "."...
%                             + tilt_stack_ali_ext;
%                         command = "tilt -InputProjections " + ctf_corrected_stack_destination...
%                             + " -OutputFile " + ctf_corrected_tomogram_destination...
%                             + " -TILTFILE " + tilt_file_destination...
%                             + " -THICKNESS " + obj.configuration.reconstruction_thickness / obj.configuration.aligned_stack_binning;
%                         if obj.configuration.set_up.gpu > 0
%                                 command = command + " -UseGPU " + num2str(obj.configuration.set_up.gpu);
%                         end
%                         executeCommand(command, false, obj.log_file_id);
%                         % TODO: if time and motivation implement exclude views by
%                         % parametrization not by truncation
%                         %                + " -EXCLUDELIST2 $EXCLUDEVIEWS");
%                         ctf_corrected_rotated_tomogram_destination = tilt_stack_ali_path + "_"...
%                             + obj.configuration.ctf_corrected_stack_suffix + "_"...
%                             + obj.configuration.tomogram_suffix + "."...
%                             + tilt_stack_ali_ext;
%                         executeCommand("trimvol -rx " + ctf_corrected_tomogram_destination...
%                             + " " + ctf_corrected_rotated_tomogram_destination, false, obj.log_file_id);
%                         if obj.configuration.generate_exact_filtered_tomograms == true
%                             disp("INFO: tomograms with exact filter (size: " + obj.configuration.exact_filter_size + ") will be generated.");
%                             ctf_corrected_exact_filtered_tomogram_destination = tilt_stack_ali_path + "_"...
%                                 + obj.configuration.ctf_corrected_stack_suffix + "_"...
%                                 + obj.configuration.exact_filter_suffix + "_"...
%                                 + obj.configuration.tomogram_suffix + "."...
%                                 + tilt_stack_ali_ext;
%                             command = "tilt -InputProjections " + ctf_corrected_stack_destination...
%                                 + " -OutputFile " + ctf_corrected_exact_filtered_tomogram_destination...
%                                 + " -TILTFILE " + tilt_file_destination...
%                                 + " -THICKNESS " + obj.configuration.reconstruction_thickness / obj.configuration.aligned_stack_binning...
%                                 + " -ExactFilterSize " + obj.configuration.exact_filter_size;
%                             if obj.configuration.set_up.gpu > 0
%                                 command = command + " -UseGPU " + num2str(obj.configuration.set_up.gpu);
%                             end
%                             executeCommand(command, false, obj.log_file_id);
%                             % TODO: if time and motivation implement exclude views by
%                             % parametrization not by truncation
%                             %                + "-EXCLUDELIST2 $EXCLUDEVIEWS");
%                             ctf_corrected_exact_filtered_rotated_tomogram_destination = tilt_stack_ali_path + "_"...
%                                 + obj.configuration.ctf_corrected_stack_suffix + "_"...
%                                 + obj.configuration.exact_filter_suffix + "_"...
%                                 + obj.configuration.tomogram_suffix + "."...
%                                 + tilt_stack_ali_ext;
%                             executeCommand("trimvol -rx " + ctf_corrected_exact_filtered_tomogram_destination...
%                                 + " " + ctf_corrected_exact_filtered_rotated_tomogram_destination, false, obj.log_file_id);
%                         end
%                     end
%                 end
%             end
%             dynamic_configuration.global_lower_defocus_average_in_angstrom = dynamic_configuration.global_lower_defocus_average_in_angstrom / length(tilt_stacks);
%             dynamic_configuration.global_upper_defocus_average_in_angstrom = dynamic_configuration.global_upper_defocus_average_in_angstrom / length(tilt_stacks);
        end
        
        function obj = cleanUp(obj)
            if obj.configuration.execute == false && obj.configuration.keep_intermediates == false
                folder = obj.output_path + string(filesep) + obj.configuration.slice_folder;
                files = dir(folder + string(filesep) + "*");
                obj.deleteFilesOrFolders(files);
                obj.deleteFolderIfEmpty(folder);
                
%                 files(1) = [];
%                 files(1) = [];
%                 files = files(~contains({files(:).name}, ".defocus"));
%                 files = files(~contains({files(:).name}, ".log"));
%                 for i = 1:length(files)
%                     if files(i).isdir == true
%                         [success, message,message_id] = rmdir(files(i).folder + string(filesep) + files(i).name, "s");
%                     else
%                         delete(files(i).folder + string(filesep) + files(i).name);
%                     end
%                 end
            end
            obj = cleanUp@Module(obj);
        end
    end
end

