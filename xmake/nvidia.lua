target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("c++17")
    add_rules("cuda")
    set_warnings("all", "error")

    add_cugencodes("native")
    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()
