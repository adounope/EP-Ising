import turtle
def draw_neural_network(layer_x = [-250, 0, 250], nudge=False, ups_h=[False, True, False, True], ups_o=[True, False, True], text_i = '', text_o = ''):

    
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

    t3.speed(5)
    t3.hideturtle()
    t3.pensize(4)
    t3.color("#FF0000")
    
    # Node positions parameters
    #layer_x = [-250, 0, 250]  # X positions for input, hidden, output layers
    node_radius = 40
    vertical_spacing = 120
    
    # Draw connections first
    def draw_connections():
        # Input to Hidden
        for i in range(3):
            for j in range(4):
                start_x = layer_x[0] + node_radius
                start_y = (i - 0.5) * vertical_spacing
                end_x = layer_x[1] - node_radius
                end_y = (j - 1) * vertical_spacing
                t.penup()
                t.goto(start_x, start_y)
                t.pendown()
                t.goto(end_x, end_y)
        
        # Hidden to Output
        for i in range(4):
            for j in range(3):
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
        ups_i = [True, False, True]
        for i in range(3):
            y = (i - 0.5) * vertical_spacing
            t.penup()
            t.goto(layer_x[0], y - node_radius)
            t.setheading(0)
            t.pendown()
            t.color("black")
            t.circle(node_radius)
            draw_arrow(x=layer_x[0], y=y, up=ups_i[i])
            t3.penup()
            t3.goto(layer_x[0], y)
            t3.pendown()
            t3.goto(layer_x[0]-120, y)
            draw_arrow(x=layer_x[0]-120, y=y, up=ups_i[i])


        
        # Hidden nodes
        #ups_h = [False, False, True, False]
        for i in range(4):
            y = (i - 1) * vertical_spacing
            t.penup()
            t.goto(layer_x[1], y - node_radius)
            t.setheading(0)
            t.pendown()
            t.color("black")
            t.circle(node_radius)
            draw_arrow(x=layer_x[1], y=y, up=ups_h[i])
        
        # Output nodes
        #ups = [True, False, True]
        for i in range(3):
            y = (i - 0.5) * vertical_spacing
            t.penup()
            t.goto(layer_x[2], y - node_radius)
            t.setheading(0)
            t.pendown()
            t.color("black")
            t.circle(node_radius)
            draw_arrow(x=layer_x[2], y=y, up=ups_o[i])
            if nudge:
                t3.penup()
                t3.goto(layer_x[2], y)
                t3.pendown()
                t3.goto(layer_x[2]+120, y)
                draw_arrow(x=layer_x[2]+120, y=y, up=ups_o[i])



    # Execution order
    draw_connections()
    draw_nodes()
    #draw_labels()

# Setup screen
screen = turtle.Screen()
screen.setup(width=1900, height=1000)
screen.title("Simple Neural Network")
screen.bgcolor("white")
x_shift = 350

draw_neural_network(layer_x=[-200-x_shift, -x_shift, 200-x_shift], nudge=False, ups_h=[False, True, False, True], ups_o=[False, True, True])
draw_neural_network(layer_x=[-200+x_shift, +x_shift, 200+x_shift], nudge=True, ups_h=[True, False, True, True], ups_o=[False, False, True])
turtle.done()