# Directorio donde se encuentran los archivos Verilog a procesar
set verilog_dir "/home/nicolas/Desktop/Verilog2Netlist/netlists"
# Cambia "tu_directorio_de_verilog" al directorio donde tienes tus archivos .v

set part_name "xc7z020clg484-1"
set current_time [clock format [clock seconds] -format "%Y_%m_%d_%H%M%S"]
#set project_name "proyecto_$current_time"
#set project_dir "proyectos/$project_name"

set project_name_base "proyecto_$current_time"
set project_dir_base "proyectos/$project_name_base"

set i 1
set project_name $project_name_base
set project_dir $project_dir_base

while {[file exists $project_dir]} {
    set project_name "${project_name_base}_${i}"
    set project_dir "proyectos/$project_name"
    incr i
}

file mkdir $project_dir
file mkdir $project_dir/netlists

# Iterar sobre todos los archivos .v en el directorio especificado
foreach verilog_file [glob -directory $verilog_dir *.v] {

	current_project [project] 
	if {[catch {current_project } result ]} {
		puts "DEBUG: Project $projectName is not open"
		
	} else {
		puts "Project $project_name is already open"
		close_project
	}

    set top_module [file rootname [file tail $verilog_file]]
    set netlist_file "$project_dir/netlists/${top_module}_netlist.v"

    create_project $project_name $project_dir -part $part_name -force
    add_files $verilog_file
    update_compile_order -fileset sources_1
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1

    #Generar netlist .v
    open_checkpoint $project_dir/$project_name.runs/synth_1/$top_module.dcp
	#close_project
	
    write_verilog $netlist_file

    # Limpiar el proyecto para la próxima iteración
	
    close_project
    
}
file delete -force $project_dir
exit