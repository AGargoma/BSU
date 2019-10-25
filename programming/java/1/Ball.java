import java.util.Iterator;

interface Shape {

    Double SurfaceArea();

    Double Volume();
}

class Ball implements Shape, Comparable<Ball>, Iterable<Object>, Iterator<Object> {

    enum Sorting_criterion {sortByRad, sortByID}

    static private int amount = 0;
    static private Sorting_criterion sorting_criterion = Sorting_criterion.sortByRad;
    private Double radius;
    private Integer id;
    private Integer iterator_index;

    @Override
    public Iterator<Object> iterator() {
        iterator_index = 0;
        return this;
    }

    @Override
    public boolean hasNext() {
        return iterator_index < 3;
    }

    @Override
    public Object next() {
        switch (iterator_index++) {
            case 0:
                return radius;
            case 1:
                return id;
            case 2:
                return sorting_criterion;
        }
        return null;
    }


    public Ball(Double radius) {
        id = amount++;
        this.radius = radius;
    }

    public Ball(String str) {
        id = amount++;
        radius = Double.parseDouble(str);

    }

    static public void setSorting_criterion(String choice) {
        try {
            sorting_criterion = Sorting_criterion.valueOf(choice);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.out.println("Sorting criterion - radius");
            sorting_criterion = Sorting_criterion.sortByRad;
        }
    }

    public Double getRadius() {
        return radius;
    }

    public void setRadius(Double radius) {
        this.radius = radius;
    }

    @Override
    public Double SurfaceArea() {
        return 4 * Math.PI * Math.pow(radius, 2);
    }

    @Override
    public Double Volume() {
        return (4. / 3.) * Math.PI * Math.pow(radius, 3);
    }

    @Override
    public String toString() {
        return "Ball{" +
                "radius=" + radius +
                ", id=" + id +
                '}';
    }

    @Override
    public int compareTo(Ball ball) {
        switch (sorting_criterion) {
            case sortByRad:
                return this.radius.compareTo(ball.radius);
            case sortByID:
                return this.id.compareTo(ball.id);
        }
        return 0;
    }
}
