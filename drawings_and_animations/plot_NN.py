

import turtle

def draw_neural_network():
    # Setup screen
    screen = turtle.Screen()
    screen.setup(width=800, height=600)
    screen.title("Simple Neural Network")
    screen.bgcolor("white")
    
    # Create turtle
    t = turtle.Turtle()
    t2 = turtle.Turtle()
    t3 = turtle.Turtle()
    t2.speed(0)
    t2.hideturtle()
    t2.pensize(16)

    t.speed(0)
    t.hideturtle()
    t.pensize(8)

    t3.speed(0)
    t3.hideturtle()
    t3.pensize(4)
    t3.color("#FF0000")
    
    # Node positions parameters
    layer_x = [-250, 0, 250]  # X positions for input, hidden, output layers
    node_radius = 40
    vertical_spacing = 160
    
    # Draw connections first
    def draw_connections():
        # Input to Hidden
        for i in range(2):
            for j in range(3):
                start_x = layer_x[0] + node_radius
                start_y = (i - 0.5) * vertical_spacing
                end_x = layer_x[1] - node_radius
                end_y = (j - 1) * vertical_spacing
                t.penup()
                t.goto(start_x, start_y)
                t.pendown()
                t.goto(end_x, end_y)
        
        # Hidden to Output
        for i in range(3):
            for j in range(2):
                start_x = layer_x[1] + node_radius
                start_y = (i - 1) * vertical_spacing
                end_x = layer_x[2] - node_radius
                end_y = (j - 0.5) * vertical_spacing
                t.penup()
                t.goto(start_x, start_y)
                t.pendown()
                t.goto(end_x, end_y)
    def draw_arrow(x, y, up=True, x_shift=0):
        t2.penup()
        # Point turtle upwards
        if up:
            t2.goto(x-x_shift, y)
            t2.color("#E0E000")
            t2.setheading(90)
        else:
            t2.goto(x+x_shift, y)
            t2.color("#E000E0")
            t2.setheading(270)
        t2.pendown()
        t2.backward(40)
        # Draw the arrow shaft
        t2.forward(80)
        # Draw the arrowhead
        tip_length = 30
        t2.right(150)
        t2.forward(tip_length)
        t2.backward(tip_length)
        t2.left(300)
        t2.forward(tip_length)
        t2.backward(tip_length)
    # Draw nodes
    def draw_nodes():
        # Input nodes
        ups = [True, False]
        for i in range(2):
            y = (i - 0.5) * vertical_spacing
            t.penup()
            t.goto(layer_x[0], y - node_radius)
            t.setheading(0)
            t.pendown()
            t.color("black")
            t.circle(node_radius)
            draw_arrow(x=layer_x[0], y=y, up=ups[i])
            t3.penup()
            t3.goto(layer_x[0], y)
            t3.pendown()
            t3.goto(layer_x[0]-100, y)
            draw_arrow(x=layer_x[0]-100, y=y, up=ups[i])

        
        # Hidden nodes
        ups = [False, False, True]
        for i in range(3):
            y = (i - 1) * vertical_spacing
            t.penup()
            t.goto(layer_x[1], y - node_radius)
            t.setheading(0)
            t.pendown()
            t.color("black")
            t.circle(node_radius)
            draw_arrow(x=layer_x[1], y=y, up=ups[i])
        
        # Output nodes
        ups = [False, True]
        for i in range(2):
            y = (i - 0.5) * vertical_spacing
            t.penup()
            t.goto(layer_x[2], y - node_radius)
            t.setheading(0)
            t.pendown()
            t.color("black")
            t.circle(node_radius)
            draw_arrow(x=layer_x[2], y=y, up=ups[i])
            # t3.penup()
            # t3.goto(layer_x[2], y)
            # t3.pendown()
            # t3.goto(layer_x[2]+100, y)
            # draw_arrow(x=layer_x[2]+100, y=y, up=ups[i])


    # Execution order
    draw_connections()
    draw_nodes()
    #draw_labels()
    turtle.done()

draw_neural_network()
