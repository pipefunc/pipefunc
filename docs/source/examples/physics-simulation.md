---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Rocket Simulation Pipeline Example

```{try-notebook}
```

This example demonstrates a comprehensive rocket flight simulation pipeline using `pipefunc`. We'll model a two-stage rocket launch from ground to orbit, incorporating physics modeling, parameter sweeps, and analysis of various performance metrics.

## Overview

Our pipeline will:
1. Set up rocket, environment, and simulation parameters
2. Simulate flight physics including thrust, drag, and gravity
3. Calculate state evolution (position, velocity, acceleration)
4. Perform stage separation and orbital insertion checks
5. Analyze performance and generate visualizations
6. Run parameter sweeps to study design trade-offs

## Setting Up

First, let's import necessary libraries and define our data structures:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.typing import Array


# --- Data Models ---

@dataclass
class RocketParameters:
    """Parameters defining rocket characteristics"""
    diameter: float  # meters
    drag_coefficient: float
    total_length: float  # meters

    # Stage 1
    stage1_dry_mass: float  # kg
    stage1_fuel_mass: float  # kg
    stage1_engine_type: str
    stage1_isp_sl: float  # seconds
    stage1_isp_vac: float  # seconds
    stage1_thrust_sl: float  # newtons
    stage1_thrust_vac: float  # newtons
    stage1_burn_time: float  # seconds

    # Stage 2
    stage2_dry_mass: float  # kg
    stage2_fuel_mass: float  # kg
    stage2_engine_type: str
    stage2_isp_vac: float  # seconds
    stage2_thrust_vac: float  # newtons
    stage2_burn_time: float  # seconds

    # Payload
    payload_mass: float  # kg


@dataclass
class LaunchConditions:
    """Environmental and initial conditions for launch"""
    launch_altitude: float  # meters above sea level
    launch_latitude: float  # degrees
    launch_longitude: float  # degrees
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    temperature: float  # Kelvin
    pressure: float  # Pascal


@dataclass
class SimulationConfig:
    """Configuration for the simulation run"""
    time_step: float  # seconds
    max_time: float  # seconds
    target_altitude: float  # meters
    target_velocity: float  # m/s


@dataclass
class FlightState:
    """State of the rocket at a point in time"""
    time: float  # seconds
    altitude: float  # meters
    velocity: float  # m/s
    acceleration: float  # m/s²
    mass: float  # kg
    thrust: float  # newtons
    drag: float  # newtons
    stage: int  # current stage (1 or 2)
    fuel_remaining: float  # kg
```

## Initialization Functions

Now let's define functions to set up our simulation:

```{code-cell} ipython3
# --- Initialization Functions ---

@pipefunc(output_name="rocket")
def create_rocket(
    diameter: float = 3.7,
    drag_coefficient: float = 0.3,
    total_length: float = 70.0,
    stage1_dry_mass: float = 22200.0,
    stage1_fuel_mass: float = 395700.0,
    stage1_engine_type: str = "Merlin 1D",
    stage1_isp_sl: float = 282.0,
    stage1_isp_vac: float = 311.0,
    stage1_thrust_sl: float = 7607000.0,
    stage1_thrust_vac: float = 8227000.0,
    stage1_burn_time: float = 162.0,
    stage2_dry_mass: float = 4000.0,
    stage2_fuel_mass: float = 92670.0,
    stage2_engine_type: str = "Merlin 1D Vacuum",
    stage2_isp_vac: float = 348.0,
    stage2_thrust_vac: float = 934000.0,
    stage2_burn_time: float = 397.0,
    payload_mass: float = 13000.0
) -> RocketParameters:
    """Create a rocket configuration with default values similar to a Falcon 9"""
    return RocketParameters(
        diameter=diameter,
        drag_coefficient=drag_coefficient,
        total_length=total_length,
        stage1_dry_mass=stage1_dry_mass,
        stage1_fuel_mass=stage1_fuel_mass,
        stage1_engine_type=stage1_engine_type,
        stage1_isp_sl=stage1_isp_sl,
        stage1_isp_vac=stage1_isp_vac,
        stage1_thrust_sl=stage1_thrust_sl,
        stage1_thrust_vac=stage1_thrust_vac,
        stage1_burn_time=stage1_burn_time,
        stage2_dry_mass=stage2_dry_mass,
        stage2_fuel_mass=stage2_fuel_mass,
        stage2_engine_type=stage2_engine_type,
        stage2_isp_vac=stage2_isp_vac,
        stage2_thrust_vac=stage2_thrust_vac,
        stage2_burn_time=stage2_burn_time,
        payload_mass=payload_mass
    )


@pipefunc(output_name="launch_conditions")
def create_launch_conditions(
    launch_altitude: float = 0.0,
    launch_latitude: float = 28.5729,  # Kennedy Space Center
    launch_longitude: float = -80.6490,
    wind_speed: float = 5.0,
    wind_direction: float = 45.0,
    temperature: float = 288.15,  # 15°C in Kelvin
    pressure: float = 101325.0  # standard atmospheric pressure (Pa)
) -> LaunchConditions:
    """Create launch conditions with default values for Kennedy Space Center"""
    return LaunchConditions(
        launch_altitude=launch_altitude,
        launch_latitude=launch_latitude,
        launch_longitude=launch_longitude,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        temperature=temperature,
        pressure=pressure
    )


@pipefunc(output_name="sim_config")
def create_simulation_config(
    time_step: float = 1.0,
    max_time: float = 600.0,
    target_altitude: float = 200000.0,  # 200 km
    target_velocity: float = 7800.0,  # orbital velocity
) -> SimulationConfig:
    """Create simulation configuration with default values"""
    return SimulationConfig(
        time_step=time_step,
        max_time=max_time,
        target_altitude=target_altitude,
        target_velocity=target_velocity
    )


@pipefunc(output_name="initial_state")
def create_initial_state(
    rocket: RocketParameters,
    launch_conditions: LaunchConditions
) -> FlightState:
    """Create the initial flight state based on rocket and launch conditions"""
    return FlightState(
        time=0.0,
        altitude=launch_conditions.launch_altitude,
        velocity=0.0,
        acceleration=0.0,
        mass=rocket.stage1_dry_mass + rocket.stage1_fuel_mass +
             rocket.stage2_dry_mass + rocket.stage2_fuel_mass +
             rocket.payload_mass,
        thrust=0.0,
        drag=0.0,
        stage=1,
        fuel_remaining=rocket.stage1_fuel_mass
    )


@pipefunc(output_name="time_points", mapspec="... -> time_points[t]", cache=True)
def generate_time_points(sim_config: SimulationConfig) -> np.ndarray:
    """Generate array of time points for the simulation"""
    return np.arange(0, sim_config.max_time, sim_config.time_step)
```

## Physics Modeling Functions

These functions calculate physical forces and effects:

```{code-cell} ipython3
# --- Physics Modeling Functions ---

@pipefunc(output_name="air_density", cache=True)
def calculate_air_density(altitude: float, temperature: float = 288.15) -> float:
    """
    Calculate air density based on altitude using barometric formula.

    Args:
        altitude: Current altitude in meters
        temperature: Temperature in Kelvin (default: 288.15K / 15°C)

    Returns:
        Air density in kg/m³
    """
    # Constants
    P0 = 101325  # sea level standard pressure (Pa)
    T0 = temperature  # sea level standard temperature (K)
    g = 9.80665  # gravitational acceleration (m/s²)
    R = 8.31447  # universal gas constant (J/(mol·K))
    M = 0.0289644  # molar mass of Earth's air (kg/mol)

    # Simplified barometric formula
    if altitude < 11000:  # troposphere
        T = T0 - 0.0065 * altitude
        P = P0 * (T/T0) ** (g*M/(R*0.0065))
    else:  # simplified for higher altitudes
        T = T0 - 0.0065 * 11000
        P = P0 * (T/T0) ** (g*M/(R*0.0065)) * np.exp(-g*M*(altitude-11000)/(R*T))

    # Ideal gas law
    density = P*M/(R*T)

    return density


@pipefunc(output_name="gravity", cache=True)
def calculate_gravity(altitude: float) -> float:
    """
    Calculate gravitational acceleration at a given altitude.

    Args:
        altitude: Current altitude in meters

    Returns:
        Gravitational acceleration in m/s²
    """
    EARTH_RADIUS = 6371000  # meters
    EARTH_MASS = 5.972e24  # kg
    G = 6.67430e-11  # gravitational constant

    # Calculate distance from Earth's center
    distance = EARTH_RADIUS + altitude

    # Calculate gravity using Newton's law of gravitation
    gravity = G * EARTH_MASS / (distance ** 2)

    return gravity


@pipefunc(output_name="thrust")
def calculate_thrust(
    rocket: RocketParameters,
    flight_state: FlightState,
    altitude: float
) -> float:
    """
    Calculate thrust based on current stage and altitude.

    Args:
        rocket: Rocket parameters
        flight_state: Current flight state
        altitude: Current altitude in meters

    Returns:
        Thrust in newtons
    """
    # If fuel is depleted, no thrust
    if flight_state.fuel_remaining <= 0:
        return 0.0

    # Calculate thrust based on stage and altitude
    if flight_state.stage == 1:
        # Linear interpolation between sea level and vacuum thrust based on altitude
        # This is a simplified model of how thrust varies with altitude
        air_density = calculate_air_density(altitude)
        sea_level_density = calculate_air_density(0)
        density_ratio = max(0, min(1, air_density / sea_level_density))

        thrust = (rocket.stage1_thrust_sl * density_ratio +
                 rocket.stage1_thrust_vac * (1 - density_ratio))
    else:  # Stage 2
        thrust = rocket.stage2_thrust_vac

    return thrust


@pipefunc(output_name="drag_force")
def calculate_drag(
    rocket: RocketParameters,
    flight_state: FlightState,
    altitude: float
) -> float:
    """
    Calculate aerodynamic drag.

    Args:
        rocket: Rocket parameters
        flight_state: Current flight state
        altitude: Current altitude in meters

    Returns:
        Drag force in newtons
    """
    # Calculate frontal area
    frontal_area = np.pi * (rocket.diameter/2)**2

    # Calculate air density
    air_density = calculate_air_density(altitude)

    # If velocity is 0 or we're in vacuum, drag is 0
    if flight_state.velocity <= 0 or air_density <= 0:
        return 0.0

    # Calculate drag force
    drag = 0.5 * air_density * (flight_state.velocity**2) * rocket.drag_coefficient * frontal_area

    return drag


@pipefunc(output_name="mass_flow_rate")
def calculate_mass_flow_rate(
    rocket: RocketParameters,
    flight_state: FlightState
) -> float:
    """
    Calculate mass flow rate based on thrust and specific impulse.

    Args:
        rocket: Rocket parameters
        flight_state: Current flight state

    Returns:
        Mass flow rate in kg/s
    """
    # If no thrust, no fuel consumption
    if flight_state.thrust <= 0:
        return 0.0

    # Calculate Isp based on stage and altitude
    if flight_state.stage == 1:
        # Linear interpolation between sea level and vacuum Isp based on altitude
        air_density = calculate_air_density(flight_state.altitude)
        sea_level_density = calculate_air_density(0)
        density_ratio = max(0, min(1, air_density / sea_level_density))

        isp = rocket.stage1_isp_sl * density_ratio + rocket.stage1_isp_vac * (1 - density_ratio)
    else:  # Stage 2
        isp = rocket.stage2_isp_vac

    # Standard gravity for Isp calculations
    g0 = 9.80665  # m/s²

    # Calculate mass flow rate using the rocket equation
    mass_flow_rate = flight_state.thrust / (isp * g0)

    return mass_flow_rate


@pipefunc(output_name="stage_status")
def check_staging(
    rocket: RocketParameters,
    flight_state: FlightState,
    time_step: float,
    mass_flow_rate: float
) -> Tuple[int, float]:
    """
    Check if staging should occur based on fuel remaining.

    Args:
        rocket: Rocket parameters
        flight_state: Current flight state
        time_step: Simulation time step in seconds
        mass_flow_rate: Current mass flow rate in kg/s

    Returns:
        Tuple of (new_stage, new_fuel_remaining)
    """
    # Calculate fuel consumed this step
    fuel_consumed = mass_flow_rate * time_step
    new_fuel_remaining = flight_state.fuel_remaining - fuel_consumed

    # Check if we need to stage
    if flight_state.stage == 1 and new_fuel_remaining <= 0:
        # Stage separation - transition to stage 2
        return 2, rocket.stage2_fuel_mass
    elif flight_state.stage == 2 and new_fuel_remaining <= 0:
        # Stage 2 fuel depleted
        return 2, 0.0
    else:
        # Continue with current stage
        return flight_state.stage, max(0, new_fuel_remaining)


@pipefunc(output_name="acceleration")
def calculate_acceleration(
    rocket: RocketParameters,
    flight_state: FlightState,
    thrust: float,
    drag_force: float,
    gravity_force: float
) -> float:
    """
    Calculate acceleration based on forces.

    Args:
        rocket: Rocket parameters
        flight_state: Current flight state
        thrust: Current thrust in newtons
        drag_force: Current drag in newtons
        gravity_force: Current gravity force in newtons

    Returns:
        Acceleration in m/s²
    """
    # Sum of forces
    net_force = thrust - drag_force - gravity_force

    # Newton's Second Law (F = ma)
    acceleration = net_force / flight_state.mass

    return acceleration
```

## Flight Simulation Functions

Here's the core flight simulation logic:

```{code-cell} ipython3
# --- Flight Simulation Functions ---

@pipefunc(output_name="flight_state", mapspec="time_points[t] -> flight_state[t]")
def simulate_timestep(
    time_point: float,
    rocket: RocketParameters,
    sim_config: SimulationConfig,
    launch_conditions: LaunchConditions,
    initial_state: FlightState = None,
    prev_state: Optional[FlightState] = None
) -> FlightState:
    """
    Simulate a single timestep of the rocket flight.

    Args:
        time_point: Current time in seconds
        rocket: Rocket parameters
        sim_config: Simulation configuration
        launch_conditions: Launch conditions
        initial_state: Initial flight state (for t=0)
        prev_state: Previous flight state (for t>0)

    Returns:
        Updated flight state
    """
    # For the first time point, use the initial state
    if time_point == 0 or prev_state is None:
        return initial_state

    # Calculate thrust based on current stage and altitude
    thrust = calculate_thrust(rocket, prev_state, prev_state.altitude)

    # Calculate drag force
    drag = calculate_drag(rocket, prev_state, prev_state.altitude)

    # Calculate gravity
    gravity_acc = calculate_gravity(prev_state.altitude)
    gravity_force = prev_state.mass * gravity_acc

    # Calculate mass flow rate
    mass_flow_rate = calculate_mass_flow_rate(rocket, prev_state)

    # Check if staging occurs
    new_stage, new_fuel_remaining = check_staging(
        rocket, prev_state, sim_config.time_step, mass_flow_rate
    )

    # Calculate new mass
    fuel_consumed = prev_state.fuel_remaining - new_fuel_remaining
    new_mass = prev_state.mass - fuel_consumed

    # If staging occurred, remove stage 1 mass
    if prev_state.stage == 1 and new_stage == 2:
        new_mass -= rocket.stage1_dry_mass

    # Calculate acceleration
    acceleration = calculate_acceleration(rocket, prev_state, thrust, drag, gravity_force)

    # Update velocity (simple Euler integration)
    new_velocity = prev_state.velocity + acceleration * sim_config.time_step

    # Update altitude (simple Euler integration)
    new_altitude = prev_state.altitude + new_velocity * sim_config.time_step

    # Create new flight state
    new_state = FlightState(
        time=time_point,
        altitude=new_altitude,
        velocity=new_velocity,
        acceleration=acceleration,
        mass=new_mass,
        thrust=thrust,
        drag=drag,
        stage=new_stage,
        fuel_remaining=new_fuel_remaining
    )

    return new_state
```

## Analysis Functions

Functions to analyze the simulation results:

```{code-cell} ipython3
# --- Analysis Functions ---

@pipefunc(output_name="max_altitude")
def find_max_altitude(flight_states: Array[FlightState]) -> float:
    """Find the maximum altitude reached during flight"""
    return max(state.altitude for state in flight_states)


@pipefunc(output_name="max_velocity")
def find_max_velocity(flight_states: Array[FlightState]) -> float:
    """Find the maximum velocity reached during flight"""
    return max(state.velocity for state in flight_states)


@pipefunc(output_name="max_acceleration")
def find_max_acceleration(flight_states: Array[FlightState]) -> float:
    """Find the maximum acceleration reached during flight"""
    return max(state.acceleration for state in flight_states)


@pipefunc(output_name="time_to_space")
def calculate_time_to_space(flight_states: Array[FlightState]) -> float:
    """Calculate time to reach space (100 km altitude)"""
    for state in flight_states:
        if state.altitude >= 100000:  # 100 km in meters
            return state.time
    return float('inf')  # Did not reach space


@pipefunc(output_name="stage_separation_data")
def analyze_stage_separation(flight_states: Array[FlightState]) -> Dict:
    """Analyze conditions at stage separation"""
    for i, state in enumerate(flight_states):
        if i > 0 and state.stage == 2 and flight_states[i-1].stage == 1:
            return {
                "time": state.time,
                "altitude": state.altitude,
                "velocity": state.velocity,
                "acceleration": state.acceleration
            }
    return {
        "time": None,
        "altitude": None,
        "velocity": None,
        "acceleration": None
    }


@pipefunc(output_name="orbital_status")
def check_orbital_insertion(
    flight_states: Array[FlightState],
    sim_config: SimulationConfig
) -> Dict:
    """Check if the rocket reached orbital insertion parameters"""
    # Get final state
    final_state = flight_states[-1]

    # Check against target parameters
    altitude_achieved = final_state.altitude >= sim_config.target_altitude
    velocity_achieved = final_state.velocity >= sim_config.target_velocity

    return {
        "success": altitude_achieved and velocity_achieved,
        "final_altitude": final_state.altitude,
        "final_velocity": final_state.velocity,
        "altitude_achieved": altitude_achieved,
        "velocity_achieved": velocity_achieved,
        "altitude_percent": (final_state.altitude / sim_config.target_altitude) * 100,
        "velocity_percent": (final_state.velocity / sim_config.target_velocity) * 100
    }
```

## Visualization Functions

Functions to create plots of the simulation results:

```{code-cell} ipython3
# --- Visualization Functions ---

@pipefunc(output_name="trajectory_plot")
def plot_trajectory(flight_states: Array[FlightState]) -> plt.Figure:
    """Generate a plot of the rocket's trajectory"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    times = [state.time for state in flight_states]
    altitudes = [state.altitude / 1000 for state in flight_states]  # Convert to km

    # Plot the trajectory
    ax.plot(times, altitudes, 'b-', linewidth=2)

    # Mark stage separation if it occurred
    for i in range(1, len(flight_states)):
        if flight_states[i].stage > flight_states[i-1].stage:
            ax.axvline(x=flight_states[i].time, color='r', linestyle='--', label='Stage Separation')
            ax.plot(flight_states[i].time, flight_states[i].altitude / 1000, 'ro')
            break

    # Mark space boundary (Kármán line - 100 km)
    ax.axhline(y=100, color='g', linestyle='--', label='Space (100 km)')

    # Add labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Rocket Trajectory')
    ax.grid(True)
    ax.legend()

    return fig


@pipefunc(output_name="performance_plots")
def plot_performance_metrics(flight_states: Array[FlightState]) -> plt.Figure:
    """Generate plots of various performance metrics"""
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Extract data
    times = [state.time for state in flight_states]
    velocities = [state.velocity for state in flight_states]
    accelerations = [state.acceleration for state in flight_states]
    masses = [state.mass for state in flight_states]

    # Plot velocity
    axs[0].plot(times, velocities, 'b-', linewidth=2)
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].set_title('Velocity vs Time')
    axs[0].grid(True)

    # Plot acceleration
    axs[1].plot(times, accelerations, 'r-', linewidth=2)
    axs[1].set_ylabel('Acceleration (m/s²)')
    axs[1].set_title('Acceleration vs Time')
    axs[1].grid(True)

    # Plot mass
    axs[2].plot(times, masses, 'g-', linewidth=2)
    axs[2].set_ylabel('Mass (kg)')
    axs[2].set_title('Mass vs Time')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].grid(True)

    # Mark stage separation if it occurred
    for i in range(1, len(flight_states)):
        if flight_states[i].stage > flight_states[i-1].stage:
            for ax in axs:
                ax.axvline(x=flight_states[i].time, color='k', linestyle='--', label='Stage Separation')
            break

    fig.tight_layout()
    return fig


@pipefunc(output_name="summary_report")
def generate_mission_summary(
    rocket: RocketParameters,
    launch_conditions: LaunchConditions,
    sim_config: SimulationConfig,
    max_altitude: float,
    max_velocity: float,
    max_acceleration: float,
    time_to_space: float,
    stage_separation_data: Dict,
    orbital_status: Dict
) -> str:
    """Generate a comprehensive mission summary"""
    summary = "===== ROCKET MISSION SUMMARY =====\n\n"

    # Rocket configuration
    summary += "ROCKET CONFIGURATION:\n"
    summary += f"- Diameter: {rocket.diameter} m\n"
    summary += f"- Total Length: {rocket.total_length} m\n"
    summary += f"- Drag Coefficient: {rocket.drag_coefficient}\n"
    summary += f"- Stage 1 Engine: {rocket.stage1_engine_type}\n"
    summary += f"- Stage 2 Engine: {rocket.stage2_engine_type}\n"
    summary += f"- Payload Mass: {rocket.payload_mass} kg\n\n"

    # Launch conditions
    summary += "LAUNCH CONDITIONS:\n"
    summary += f"- Launch Site: ({launch_conditions.launch_latitude}°, {launch_conditions.launch_longitude}°)\n"
    summary += f"- Launch Altitude: {launch_conditions.launch_altitude} m\n"
    summary += f"- Wind Speed: {launch_conditions.wind_speed} m/s at {launch_conditions.wind_direction}°\n"
    summary += f"- Temperature: {launch_conditions.temperature - 273.15:.1f}°C\n\n"

    # Mission parameters
    summary += "MISSION PARAMETERS:\n"
    summary += f"- Target Altitude: {sim_config.target_altitude/1000} km\n"
    summary += f"- Target Velocity: {sim_config.target_velocity} m/s\n\n"

    # Mission results
    summary += "MISSION RESULTS:\n"
    summary += f"- Maximum Altitude: {max_altitude/1000:.2f} km\n"
    summary += f"- Maximum Velocity: {max_velocity:.2f} m/s\n"
    summary += f"- Maximum Acceleration: {max_acceleration:.2f} m/s² ({max_acceleration/9.81:.2f} G)\n"
    summary += f"- Time to Space (100 km): {time_to_space:.2f} seconds\n\n"

    # Stage separation
    if stage_separation_data["time"] is not None:
        summary += "STAGE SEPARATION:\n"
        summary += f"- Time: {stage_separation_data['time']:.2f} seconds\n"
        summary += f"- Altitude: {stage_separation_data['altitude']/1000:.2f} km\n"
        summary += f"- Velocity: {stage_separation_data['velocity']:.2f} m/s\n\n"
    else:
        summary += "STAGE SEPARATION: Did not occur\n\n"

    # Orbital insertion
    summary += "ORBITAL INSERTION:\n"
    if orbital_status["success"]:
        summary += "- STATUS: SUCCESS ✓\n"
    else:
        summary += "- STATUS: FAILED ✗\n"

    summary += f"- Final Altitude: {orbital_status['final_altitude']/1000:.2f} km ({orbital_status['altitude_percent']:.1f}% of target)\n"
    summary += f"- Final Velocity: {orbital_status['final_velocity']:.2f} m/s ({orbital_status['velocity_percent']:.1f}% of target)\n"

    return summary
```

## Parameter Sweep Functions

These functions demonstrate how to use pipefunc to systematically explore design parameters:

```{code-cell} ipython3
# --- Parameter Sweep Functions ---

@pipefunc(output_name="sweep_result", mapspec="payload_masses[p] -> sweep_result[p]")
def analyze_payload_impact(
    payload_mass: float,
    rocket: RocketParameters,
    launch_conditions: LaunchConditions,
    sim_config: SimulationConfig
) -> Dict:
    """Analyze the impact of payload mass on maximum altitude"""
    # Create a modified rocket with the specified payload
    modified_rocket = RocketParameters(
        diameter=rocket.diameter,
        drag_coefficient=rocket.drag_coefficient,
        total_length=rocket.total_length,
        stage1_dry_mass=rocket.stage1_dry_mass,
        stage1_fuel_mass=rocket.stage1_fuel_mass,
        stage1_engine_type=rocket.stage1_engine_type,
        stage1_isp_sl=rocket.stage1_isp_sl,
        stage1_isp_vac=rocket.stage1_isp_vac,
        stage1_thrust_sl=rocket.stage1_thrust_sl,
        stage1_thrust_vac=rocket.stage1_thrust_vac,
        stage1_burn_time=rocket.stage1_burn_time,
        stage2_dry_mass=rocket.stage2_dry_mass,
        stage2_fuel_mass=rocket.stage2_fuel_mass,
        stage2_engine_type=rocket.stage2_engine_type,
        stage2_isp_vac=rocket.stage2_isp_vac,
        stage2_thrust_vac=rocket.stage2_thrust_vac,
        stage2_burn_time=rocket.stage2_burn_time,
        payload_mass=payload_mass
    )

    # Create initial state with the modified rocket
    initial_state = create_initial_state(modified_rocket, launch_conditions)

    # Run a simplified simulation (fewer time points)
    time_points = np.arange(0, sim_config.max_time, sim_config.time_step * 5)  # Use larger time step

    # Manual simulation loop for this analysis (simplified)
    states = [initial_state]
    for t in time_points[1:]:
        next_state = simulate_timestep(
            time_point=t,
            rocket=modified_rocket,
            sim_config=sim_config,
            launch_conditions=launch_conditions,
            prev_state=states[-1]
        )
        states.append(next_state)

        # Early termination if we're falling back to Earth
        if len(states) > 2 and states[-1].altitude < states[-2].altitude and states[-1].velocity < 0:
            break

    # Extract key metrics
    max_alt = max(state.altitude for state in states)
    max_vel = max(state.velocity for state in states)
    final_state = states[-1]

    # Return performance metrics as a dictionary
    return {
        "payload_mass": payload_mass,
        "max_altitude": max_alt,
        "max_velocity": max_vel,
        "final_altitude": final_state.altitude,
        "final_velocity": final_state.velocity,
        "orbit_achieved": (
            final_state.altitude >= sim_config.target_altitude and
            final_state.velocity >= sim_config.target_velocity
        )
    }


@pipefunc(output_name="multi_sweep_result",
          mapspec="payload_masses[p], thrust_multipliers[t], isp_multipliers[i] -> multi_sweep_result[p, t, i]")
def analyze_multi_parameter_impact(
    payload_mass: float,
    thrust_multiplier: float,
    isp_multiplier: float,
    rocket: RocketParameters,
    launch_conditions: LaunchConditions,
    sim_config: SimulationConfig
) -> Dict:
    """
    Analyze the combined impact of payload mass, thrust, and specific impulse
    on rocket performance.

    Args:
        payload_mass: Payload mass in kg
        thrust_multiplier: Factor to multiply all thrust values by
        isp_multiplier: Factor to multiply all specific impulse values by
        rocket: Base rocket parameters
        launch_conditions: Launch conditions
        sim_config: Simulation configuration

    Returns:
        Dictionary of performance metrics
    """
    # Create a modified rocket with the specified parameters
    modified_rocket = RocketParameters(
        diameter=rocket.diameter,
        drag_coefficient=rocket.drag_coefficient,
        total_length=rocket.total_length,
        stage1_dry_mass=rocket.stage1_dry_mass,
        stage1_fuel_mass=rocket.stage1_fuel_mass,
        stage1_engine_type=rocket.stage1_engine_type,
        stage1_isp_sl=rocket.stage1_isp_sl * isp_multiplier,
        stage1_isp_vac=rocket.stage1_isp_vac * isp_multiplier,
        stage1_thrust_sl=rocket.stage1_thrust_sl * thrust_multiplier,
        stage1_thrust_vac=rocket.stage1_thrust_vac * thrust_multiplier,
        stage1_burn_time=rocket.stage1_burn_time,
        stage2_dry_mass=rocket.stage2_dry_mass,
        stage2_fuel_mass=rocket.stage2_fuel_mass,
        stage2_engine_type=rocket.stage2_engine_type,
        stage2_isp_vac=rocket.stage2_isp_vac * isp_multiplier,
        stage2_thrust_vac=rocket.stage2_thrust_vac * thrust_multiplier,
        stage2_burn_time=rocket.stage2_burn_time,
        payload_mass=payload_mass
    )

    # Similar simplified simulation as in analyze_payload_impact
    initial_state = create_initial_state(modified_rocket, launch_conditions)

    # Run a simplified simulation
    time_points = np.arange(0, sim_config.max_time, sim_config.time_step * 5)

    # Manual simulation loop
    states = [initial_state]
    for t in time_points[1:]:
        next_state = simulate_timestep(
            time_point=t,
            rocket=modified_rocket,
            sim_config=sim_config,
            launch_conditions=launch_conditions,
            prev_state=states[-1]
        )
        states.append(next_state)

        # Early termination if we're falling back to Earth
        if len(states) > 2 and states[-1].altitude < states[-2].altitude and states[-1].velocity < 0:
            break

    # Extract key metrics
    max_alt = max(state.altitude for state in states)
    max_vel = max(state.velocity for state in states)
    final_state = states[-1]

    # Return performance metrics
    return {
        "payload_mass": payload_mass,
        "thrust_multiplier": thrust_multiplier,
        "isp_multiplier": isp_multiplier,
        "max_altitude": max_alt,
        "max_velocity": max_vel,
        "final_altitude": final_state.altitude,
        "final_velocity": final_state.velocity,
        "orbit_achieved": (
            final_state.altitude >= sim_config.target_altitude and
            final_state.velocity >= sim_config.target_velocity
        )
    }


@pipefunc(output_name="payload_sweep_plot")
def plot_payload_sweep(sweep_results: Array[Dict]) -> plt.Figure:
    """Plot the results of the payload sweep analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract data
    payload_masses = [result["payload_mass"] for result in sweep_results]
    max_altitudes = [result["max_altitude"] / 1000 for result in sweep_results]  # km
    max_velocities = [result["max_velocity"] / 1000 for result in sweep_results]  # km/s

    # Plot altitude vs payload mass
    ax1.plot(payload_masses, max_altitudes, 'b-', marker='o')
    ax1.set_xlabel('Payload Mass (kg)')
    ax1.set_ylabel('Maximum Altitude (km)')
    ax1.set_title('Maximum Altitude vs Payload Mass')
    ax1.grid(True)

    # Plot velocity vs payload mass
    ax2.plot(payload_masses, max_velocities, 'r-', marker='o')
    ax2.set_xlabel('Payload Mass (kg)')
    ax2.set_ylabel('Maximum Velocity (km/s)')
    ax2.set_title('Maximum Velocity vs Payload Mass')
    ax2.grid(True)

    fig.tight_layout()
    return fig


@pipefunc(output_name="multi_sweep_plot")
def plot_multi_sweep(multi_sweep_results: Array[Dict]) -> Dict[str, plt.Figure]:
    """Generate plots from the multi-parameter sweep results"""
    # Reshape the results for easier analysis
    results_dict = {}
    for result in multi_sweep_results.flatten():
        payload = result["payload_mass"]
        thrust = result["thrust_multiplier"]
        isp = result["isp_multiplier"]

        key = (payload, thrust, isp)
        results_dict[key] = {
            "max_altitude": result["max_altitude"],
            "max_velocity": result["max_velocity"],
            "orbit_achieved": result["orbit_achieved"]
        }

    # Create plots for each payload mass
    unique_payloads = sorted({result["payload_mass"] for result in multi_sweep_results.flatten()})
    unique_thrusts = sorted({result["thrust_multiplier"] for result in multi_sweep_results.flatten()})
    unique_isps = sorted({result["isp_multiplier"] for result in multi_sweep_results.flatten()})

    plots = {}

    # For each payload, create a 2D heatmap of altitude vs thrust and ISP
    for payload in unique_payloads:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a grid for the heatmap
        data = np.zeros((len(unique_thrusts), len(unique_isps)))
        orbit_achieved = np.zeros((len(unique_thrusts), len(unique_isps)), dtype=bool)

        for i, thrust in enumerate(unique_thrusts):
            for j, isp in enumerate(unique_isps):
                key = (payload, thrust, isp)
                if key in results_dict:
                    data[i, j] = results_dict[key]["max_altitude"] / 1000  # km
                    orbit_achieved[i, j] = results_dict[key]["orbit_achieved"]

        # Create heatmap
        im = ax.imshow(data, origin='lower', aspect='auto', cmap='viridis')

        # Mark orbit achieved cells
        for i in range(len(unique_thrusts)):
            for j in range(len(unique_isps)):
                if orbit_achieved[i, j]:
                    ax.plot(j, i, 'ro', markersize=4)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Maximum Altitude (km)')

        # Set tick labels
        ax.set_xticks(np.arange(len(unique_isps)))
        ax.set_yticks(np.arange(len(unique_thrusts)))
        ax.set_xticklabels([f"{x:.1f}" for x in unique_isps])
        ax.set_yticklabels([f"{x:.1f}" for x in unique_thrusts])

        # Add labels and title
        ax.set_xlabel('ISP Multiplier')
        ax.set_ylabel('Thrust Multiplier')
        ax.set_title(f'Maximum Altitude for Payload Mass = {payload} kg')

        # Add a note about red dots
        ax.text(0.05, 0.05, "Red dots: Orbit achieved", transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

        plots[f"payload_{payload}"] = fig

    return plots
```

## Constructing the Full Pipeline

Now we'll connect all our functions into a complete pipeline:

```{code-cell} ipython3
# --- Create the Main Simulation Pipeline ---
main_simulation_pipeline = Pipeline([
    # Initialization
    create_rocket,
    create_launch_conditions,
    create_simulation_config,
    create_initial_state,
    generate_time_points,

    # Flight simulation
    simulate_timestep,

    # Analysis
    find_max_altitude,
    find_max_velocity,
    find_max_acceleration,
    calculate_time_to_space,
    analyze_stage_separation,
    check_orbital_insertion,

    # Visualization
    plot_trajectory,
    plot_performance_metrics,
    generate_mission_summary
])

# --- Create the Parameter Sweep Pipeline ---
parameter_sweep_pipeline = Pipeline([
    # Base setup
    create_rocket,
    create_launch_conditions,
    create_simulation_config,

    # Parameter sweep
    analyze_payload_impact,
    plot_payload_sweep,

    # Multi-parameter sweep
    analyze_multi_parameter_impact,
    plot_multi_sweep
])

# Visualize the main simulation pipeline
main_simulation_pipeline.visualize(backend="graphviz")
```

## Running the Simulation

Now we'll run the simulation and analyze the results:

```{code-cell} ipython3
# Run the main simulation using default values
sim_results = main_simulation_pipeline.map(
    inputs={},  # Using default values
    run_folder="rocket_simulation",
    show_progress=True
)

# Print mission summary
print(sim_results["summary_report"].output)

# Display trajectory plot
sim_results["trajectory_plot"].output
```

## Parameter Sweeps

Now let's run some parameter sweeps to analyze rocket performance under different conditions:

```{code-cell} ipython3
# Define parameter sweep inputs
sweep_inputs = {
    # Vary payload mass from 5000 kg to 20000 kg
    "payload_masses": np.linspace(5000, 20000, 7),

    # For multi-dimensional sweep, also vary thrust and ISP
    "thrust_multipliers": np.linspace(0.8, 1.2, 5),
    "isp_multipliers": np.linspace(0.9, 1.1, 5)
}

# Run parameter sweeps
sweep_results = parameter_sweep_pipeline.map(
    inputs=sweep_inputs,
    run_folder="rocket_parameter_sweeps",
    show_progress=True
)

# Display the payload sweep plot
sweep_results["payload_sweep_plot"].output

# Display one of the multi-parameter sweep plots
list(sweep_results["multi_sweep_plot"].output.values())[0]
```

## Creating a Nested Pipeline

Let's demonstrate how to use pipefunc's `nest_funcs` feature to simplify the simulation pipeline:

```{code-cell} ipython3
# Create a copy of the main simulation pipeline
nested_pipeline = main_simulation_pipeline.copy()

# Nest the flight physics functions into a single node
nested_pipeline.nest_funcs(
    {"thrust", "drag_force", "mass_flow_rate", "stage_status", "acceleration"},
    new_output_name="physics_outputs",
    function_name="flight_physics"
)

# Visualize the simplified pipeline
nested_pipeline.visualize(backend="graphviz")
```

## Using VariantPipeline for Alternative Physics Models

Let's demonstrate how to use `VariantPipeline` to offer alternative physics models:

```{code-cell} ipython3
from pipefunc import VariantPipeline

# Create a simplified physics model as an alternative
@pipefunc(output_name="air_density", variant="simple")
def calculate_air_density_simple(altitude: float) -> float:
    """Simplified exponential atmospheric model"""
    return 1.225 * np.exp(-altitude / 8500)

@pipefunc(output_name="air_density", variant="standard")
def calculate_air_density_standard(altitude: float, temperature: float = 288.15) -> float:
    """Standard atmospheric model (same as original)"""
    # Constants
    P0 = 101325  # sea level standard pressure (Pa)
    T0 = temperature  # sea level standard temperature (K)
    g = 9.80665  # gravitational acceleration (m/s²)
    R = 8.31447  # universal gas constant (J/(mol·K))
    M = 0.0289644  # molar mass of Earth's air (kg/mol)

    # Simplified barometric formula
    if altitude < 11000:  # troposphere
        T = T0 - 0.0065 * altitude
        P = P0 * (T/T0) ** (g*M/(R*0.0065))
    else:  # simplified for higher altitudes
        T = T0 - 0.0065 * 11000
        P = P0 * (T/T0) ** (g*M/(R*0.0065)) * np.exp(-g*M*(altitude-11000)/(R*T))

    # Ideal gas law
    density = P*M/(R*T)

    return density

# Create a VariantPipeline that allows switching between physics models
variant_pipeline = VariantPipeline(
    [calculate_air_density_simple, calculate_air_density_standard],
    default_variant="standard"
)

# Visualize the variant pipeline
variant_pipeline.visualize(backend="graphviz")

# Get specific variants
simple_pipeline = variant_pipeline.with_variant("simple")
standard_pipeline = variant_pipeline.with_variant("standard")

# Compare results at 30km altitude
print(f"Simple model air density at 30km: {simple_pipeline(altitude=30000):.6f} kg/m³")
print(f"Standard model air density at 30km: {standard_pipeline(altitude=30000):.6f} kg/m³")
```

## Conclusion

This example has demonstrated how pipefunc can be used to build a complex simulation pipeline that models rocket flight. We've shown:

1. **Function Composition**: Building a complex pipeline from small, focused functions
2. **Map-Reduce Operations**: Using mapspec to process time series data
3. **Parameter Sweeps**: Exploring the impact of various design parameters
4. **Visualization**: Creating plots of the results
5. **Nested Functions**: Simplifying the pipeline by grouping related operations
6. **Variant Pipelines**: Offering alternative implementations of components

The pipeline automatically determines the execution order, manages dependencies, and can be executed in parallel. All intermediate results are cached and can be reused, making it efficient for parameter exploration and design optimization.

This example can be extended to include more detailed physical models, additional design parameters, or integration with external tools like CFD simulations or structural analysis.
