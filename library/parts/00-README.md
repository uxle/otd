# Parts Library — single-material mechanical components

45 parts, each ONE material, each NAMED, each with live physics
(`ask "mass?"`, `simulate: drop / float / collapse / splash`).

## Fasteners
bolt_m6 … bolt_m16 · nut_m6 … nut_m16 · washer_m6 … washer_m12
(hex heads are real 6-sided prisms; nuts have true cut-through bores)

## The gear family (numbered by tooth count)
gear_five · gear_six · gear_seven · gear_eight · gear_nine · gear_ten ·
gear_eleven · gear_twelve · gear_pair_five_six · gear_train_five_to_twelve

## Engine parts
piston_v2 · piston_ring · piston_assembly · connecting_rod · crankshaft ·
flywheel · cam_plate · spring_coil · wedge_key · chain_link · rivet

## Transmission & hardware
pulley · bearing_bronze · axle_shaft · shaft_coupling · pipe_flange ·
handwheel · lever_arm · fan_rotor · motor_rotor · motor_assembly

## Animate any named part
    otd --png flywheel.otd spin.png --spin flywheel=90@x
    otd --png motor_assembly.otd frame.png --spin rotor --frames 8

All 45 files pass `otd --check` (see tests/parts validation sweep).
