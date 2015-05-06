require "torch"
require "lfs"

local function remove_file_if_exists(file, silent)
	if not paths.filep(file) then
		return
	end

	local success, err = os.remove(file)
	if success and not silent then
		print("Removed file `" .. file .. "`.")
	elseif not success then
		error("Failed to remove file `" .. file .. "`: " .. err .. ".")
	end
end

local function remove_empty_directory(dir, silent)
	if not paths.dirp(dir) then
		error("Could not find directory `" .. dir .. "`.")
	end

	local success, err = lfs.rmdir(dir)
	if success and not silent then
		print("Removed directory `" .. dir .. "`.")
	elseif not success then
		error("Failed to remove directory `" .. dir .. "`: " .. err .. ".")
	end
end

local function remove_directory(dir, silent)
	if not paths.dirp(dir) then
		return
	end

	for file in lfs.dir(dir) do
		local path = paths.concat(dir, file)
		if lfs.attributes(path, "mode") ~= "directory" then
			remove_file_if_exists(path, silent)
		end
	end

	remove_empty_directory(dir, silent)
	print("")
end

local function make_directory(dir, silent)
	if not paths.mkdir(dir) then
		error("Failed to create directory `" .. dir .. "`.")
	elseif not silent then
		print("Created directory `" .. dir .. "`.")
	end
end

local function create_hard_link(target, link, silent)
	local success, err = lfs.link(target, link)
	if success and not silent then
		print("Created hard link `" .. link .. "`.")
	elseif not success then
		error("Failed to create hard link `" .. link .. "`: " .. err .. ".")
	end
end

local function rename_file_if_exists(old, new, silent)
	if not paths.filep(old) then
		return
	end

	local success, err = os.rename(old, new)
	if success and not silent then
		print("Renamed file `" .. old .. "` to `" .. new .. "`.")
	elseif not success then
		print("Failed to rename file `" .. old .. "` to `" .. new .. "`: " .. err .. ".")
	end
end

local function rename_backup(backup, new)
	if not paths.filep(backup) then
		return true
	elseif paths.filep(backup) and paths.filep(new) then
		print("Both `" .. new .. "` and `" .. backup .. "` exist.")
		return false
	end

	rename_file_if_exists(backup, new)
	return true
end

local function parse_arguments()
	if not opt then
		local cmd = torch.CmdLine()
		cmd:text("Select one of the following options:")
		cmd:option("-task", "create", "create | resume | replace | evaluate")
		cmd:option("-model", "test", "The model name.")
		cmd:option("-version", "current", "current | best_train | best_test")
		cmd:option("-device", 0, "GPU device number.")
		opt = cmd:parse(arg or {})
	end

	if string.match(model_name, "^[A-Za-z0-9_]+") == nil then
		error("Invalid model name `" .. model_name .. "`.")
	end

	if opt.task == "create" then
		if paths.dirp(output_dir) then
			error("Model `" .. model_name .. "` already exists.")
		end
		make_directory(output_dir)
	elseif opt.task == "resume" or opt.task == "evaluate" then
		if not paths.dirp(output_dir) then
			error("Model `" .. model_name .. "` does not exist.")
		end
	elseif opt.task == "replace" then
		if paths.dirp(output_dir) then
			remove_directory(output_dir)
			make_directory(output_dir)
		end
	else
		error("Invalid task `" .. opt.task .. "`.")
	end
end

local function restore_backups(paths)
	-- Deal with the backup files.
	local status = true
	status = status and rename_backup(paths.cur_model_backup_fn,
		paths.cur_model_fn)
	status = status and rename_backup(paths.best_train_model_backup_fn,
		paths.best_train_model_fn)
	status = status and rename_backup(paths.best_test_model_backup_fn,
		paths.best_test_model_fn)
	status = status and rename_backup(paths.cur_train_info_backup_fn,
		paths.cur_train_info_fn)
	status = status and rename_backup(paths.best_train_train_info_backup_fn,
		paths.best_train_train_info_fn)
	status = status and rename_backup(paths.best_test_train_info_backup_fn,
		paths.best_test_train_info_fn)
	status = status and rename_backup(paths.acc_info_backup_fn,
		paths.acc_info_fn)

	if not status then
		print("Both backup and non-backup versions of certain files " ..
			"exist (see above output).")
		print("There may be data corruption.")
		print("Please carefully inspect the files, and eliminate " ..
			"the duplicates.")
		os.exit(1)
	end
end

local function deserialize(paths, model_func, train_func)
	-- Determine the files from which we are to restore the model and
	-- training info states.
	local target_model_fn = ""
	local target_train_info_fn = ""

	if opt.version == "current" then
		target_model_fn = paths.cur_model_fn
		target_train_info_fn = paths.cur_train_info_fn
	elseif opt.version == "best_train" then
		target_model_fn = paths.best_train_model_fn
		target_train_info_fn = paths.best_train_train_info_fn
	elseif opt.version == "best_test" then
		target_model_fn = paths.best_test_model_fn
		target_train_info_fn = paths.best_test_train_info_fn
	else
		error("Invalid model version `" .. opt.version .. "`.")
	end

	-- Deserialize the model and training states, or initialize new ones.
	local model_info = {}
	local train_info = {}
	local acc_info = {}

	if paths.filep(target_model_fn) then
		print("Restoring model from `" .. target_model_fn .. "`.")
		model_info = torch.load(target_model_fn)
	else
		if opt.task == "evaluate" then
			error("Model file `" .. target_model_fn .. "` not found.")
		end
		print("Creating new model.")
		model_info = model_func()
	end

	if paths.filep(target_train_info_fn) then
		print("Restoring training info from `" .. target_train_info_fn .. "`.")
		train_info = torch.load(target_train_info_fn)
	else
		if opt.task == "evaluate" then
			error("Train info file `" .. target_train_info_fn .. "` not found.")
		end
		print("Initializing training state.")
		train_info = train_func()
	end

	if paths.filep(acc_info_fn) then
		print("Restoring accuracy info from `" .. paths.acc_info_fn .. "`.")
		acc_info = torch.load(paths.acc_info_fn)
	else
		print("Initializing accuracy info.")
		acc_info = {
			best_train = 1e10,
			best_test = 1e10
		}
	end
end

local function save_train_progress(new_best, paths, info)
	print("Saving current model and training info.")
	rename_file_if_exists(paths.cur_model_fn,
		paths.cur_model_backup_fn, true)
	rename_file_if_exists(paths.cur_train_info_fn,
		paths.cur_train_info_backup_fn, true)
	torch.save(paths.cur_model_fn, info.model)
	torch.save(paths.cur_train_info_fn, info.train)

	if new_best then
		print("New best train perplexity: updating hard links.")
		rename_file_if_exists(paths.best_train_model_fn,
			paths.best_train_model_backup_fn, true)
		rename_file_if_exists(paths.best_train_train_info_fn,
			paths.best_train_train_info_backup_fn, true)
		create_hard_link(paths.cur_model_fn,
			paths.best_train_model_fn, true)
		create_hard_link(paths.cur_train_info_fn,
			paths.best_train_train_info_fn, true)
		remove_file_if_exists(paths.best_train_model_backup_fn, true)
		remove_file_if_exists(paths.best_train_train_info_backup_fn, true)

		print("Saving accuracy info.")
		rename_file_if_exists(paths.acc_info_fn,
			paths.acc_info_backup_fn, true)
		torch.save(paths.acc_info_fn, info.acc)
		remove_file_if_exists(paths.acc_info_backup_fn, true)
	end

	remove_file_if_exists(paths.cur_model_backup_fn, true)
	remove_file_if_exists(paths.cur_train_info_backup_fn, true)
end

function save_test_progress(new_best, paths, info)
	if new_best then
		print("New best test perplexity: updating hard links.")
		rename_file_if_exists(paths.best_test_model_fn,
			paths.best_test_model_backup_fn, true)
		rename_file_if_exists(paths.best_test_train_info_fn,
			paths.best_test_train_info_backup_fn, true)
		create_hard_link(paths.cur_model_fn,
			paths.best_test_model_fn, true)
		create_hard_link(paths.cur_train_info_fn,
			paths.best_test_train_info_fn, true)
		remove_file_if_exists(paths.best_test_model_backup_fn, true)
		remove_file_if_exists(paths.best_test_train_info_backup_fn, true)

		print("Saving accuracy info.")
		rename_file_if_exists(paths.acc_info_fn,
			paths.acc_info_backup_fn, true)
		torch.save(paths.acc_info_fn, info.acc)
		remove_file_if_exists(paths.acc_info_backup_fn, true)
	end
end

function start(model_func, train_func)
	local models_dir = "models"
	local model_name = opt.model
	local output_dir = paths.concat(models_dir, model_name)

	-- Define the paths to the output files for serialization.
	local paths = {
		cur_model_fn = paths.concat(output_dir,
			"cur_model.t7"),
		best_train_model_fn = paths.concat(output_dir,
			"best_train_model.t7"),
		best_test_model_fn = paths.concat(output_dir,
			"best_test_model.t7"),

		cur_train_info_fn = paths.concat(output_dir,
			"cur_train_info.t7"),
		best_train_train_info_fn = paths.concat(output_dir,
			"best_train_train_info.t7"),
		best_test_train_info_fn = paths.concat(output_dir,
			"best_test_train_info.t7"),

		cur_model_backup_fn = paths.concat(output_dir,
			"cur_model_backup.t7"),
		best_train_model_backup_fn = paths.concat(output_dir,
			"best_train_model_backup.t7"),
		best_test_model_backup_fn = paths.concat(output_dir,
			"best_test_model_backup.t7"),

		cur_train_info_backup_fn = paths.concat(
			output_dir, "cur_train_info_backup.t7"),
		best_train_train_info_backup_fn = paths.concat(
			output_dir, "best_train_train_info_backup.t7"),
		best_test_train_info_backup_fn = paths.concat(
			output_dir, "best_test_train_info_backup.t7"),

		acc_info_fn = paths.concat(output_dir,
			"acc_info.t7"),
		acc_info_backup_fn = paths.concat(output_dir,
			"acc_info_backup.t7")
	}

	print("Checking for backups.")
	restore_backups(paths)
	print("Loading data.")
	local info = deserialize(paths, model_func, train_info)

	local do_train = opt.task ~= "evaluate"
	local do_test = true
	return do_train, do_test, paths, info
end
